"""
Nonstabilizerness (magic) of the (2+1)D SU(2) hardcore-gluon lattice
gauge theory, b=0 sector.

Pipeline per g^2:
  1. build a U(1)(baryon-number)-conserving TeNPy Hamiltonian on dressed sites,
     each in a *canonical* gauge-invariant basis (simultaneous eigenbasis of
     G^2, the four leg-Casimirs T^2 and N) padded to 2^q qubits (q = 3/4/5 for
     boundary/edge/bulk sites);  DMRG ground state.
  2. gauge observables: energy density eps, electric <E^2>, magnetic <B^2>,
     average plaquette Re<B_p>.
  3. magic: stabilizer 2-Renyi entropy M_2 (exact fast-Walsh-Hadamard while the
     state fits, else an exact Pauli-completeness perfect sampler with
     parallel batched exact perfect sampling with optional final-site Rao--Blackwellization), plus the
     exact antiflatness lower bound 4F <= M_2^NL on nonlocal magic.
  4. optional Clifford (CAMPS) disentangling utilities (run_camps_gs / _warm /
     camps_reduce) are provided but the default scan uses plain DMRG.

Ground states are checkpointed (see SAVE_GS/LOAD_GS); observables and magic are
recomputed on load.

    H = m sum_j (-1)^(jx+jy) N_j                          mass
      + (g^2/2) sum_j Gamma_j                             electric
      - 1/(2g^2) sum_plaq (Tr box + h.c.)                 magnetic,  Tr box = C_pp C_mp C_pm C_mm
      + t sum_j [ -i Qd_px(j) Q_mx(j+x) + h.c. ]          hopping x   (t = 1/2)
      + t sum_j [ -(-1)^(jx+jy) Qd_py(j) Q_my(j+y) + h.c.] hopping y
      + penalties (padding leakage + link flux matching)

CONVENTIONS (must stay in sync with App. A of the paper):
  * rishon matrices carry a factor 2^(-1/4), so that U^{ab}_{j,j+mu} = zeta^a_{j,+mu} zeta^{b+}_{j+mu,-mu}
    restricted to the flux-matched link space has matrix elements +-1/sqrt(2), i.e. U coincides with the
    truncated parallel transporter of Cataldi et al., arXiv:2307.09396, Eq. (B33) (Peter--Weyl normalisation
    <1/2 m m'|U^{ab}|0> = delta_{ma} delta_{m'b}/sqrt2).
  * the corner operators C_pp, C_mp, C_pm, C_mm are the site-by-site regrouping of
    Tr(U_1 U_2 U_3^+ U_4^+); with the intra-site Jordan-Wigner ordering [matter, mx, my, px, py] this gives
        C_pp = sum_c (zeta_c P)_px (zeta_c^+)_py
        C_mp = sum_c (zeta_c^+ P)_mx P_my P_px (zeta_c)_py
        C_pm = sum_c (P zeta_c)_my (zeta_c^+)_px
        C_mm = sum_c (P zeta_c)_mx (zeta_c^+)_my
    and Tr box = C_pp(j) C_mp(j+x) C_pm(j+y) C_mm(j+x+y) EXACTLY (no extra sign).  An earlier version of this
    file had the opposite overall sign at three corners, i.e. it implemented +(1/g^2) Re Tr box; that model is
    unitarily equivalent to the theory WITHOUT staggered hopping phases and is NOT the Kogut-Susskind theory.
  * magnetic coefficient 1/g^2 (standard KS with Re Tr); NOTE Cataldi et al. use 8/g^2 in their Eq. (3)/(B45).
  * observables: <E^2> = (1/N_links) sum_links <E^2_l>,  Re<B_p> = (1/N_plaq) sum_plaq Re<Tr box>  (max 1),
    <B^2> = (1 - Re<B_p>)/2,  eps = (<H> - <H_penalty>)/N_sites.
  * run with  --selftest  to compare the 2x2 Hamiltonian against independently obtained reference energies.
"""

# Parallel batches helps in speedup. 

import os
# Set TPSRE_THREADS=4 (or 8) in the shell to benchmark threaded BLAS/SVD.
# Default stays conservative to avoid oversubscription on shared/HPC nodes.
_TPSRE_THREADS = os.environ.get("TPSRE_THREADS", "1")
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _TPSRE_THREADS)

import time, itertools, pickle, json, re, glob, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
import numpy as np
import warnings; warnings.filterwarnings("ignore")
#import matplotlib; matplotlib.use("Agg")
#import matplotlib.pyplot as plt
import datetime

from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.linalg.charges import LegCharge, ChargeInfo
from tenpy.models.lattice import Lattice as TeNPyLattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.algorithms import dmrg
import tenpy.linalg.np_conserved as npc

startTime = time.time()
print ("STARTED:" , datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S"))

# Production defaults.  Lattice, chi, mass and the g^2 list are normally given on
# the command line by the job script (su2_scan.sh), e.g.
#   python -u 2dSU2_improved.py --lattice 6x6 --chi 64 --mass 1 --no-magic --no-validate-nl
LATTICE            = "4x4"          # any 'LxxLy' label (--lattice)
G2_VALUES          = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.0, 4.0, 10.0, 100.0]   # 11-point scan (--g2)
MASS               = 1.0            # production: 1.0 and 0.2 (--mass)
CHI                = 64             # DMRG bond dimension (--chi)
MAX_SWEEPS         = 28


#  "off"    : no Cliffords.  M_2 is read directly off the DMRG MPS at full chi.
#             Assumption-free baseline, most expensive sampling.
#  "magic"  : plain U(1) DMRG (energy/observables unchanged), then a Clifford
#             disentangling circuit is applied and M_2 is read off the compressed
#             state.  SRE is Clifford-invariant so the total magic is identical to
#             "off"; the win is that sampling runs at CHI_MAGIC < CHI (cost grows
#             with chi), which is the lever for larger lattices.  Set CHI_MAGIC.
#  "gs"     : alternating CAMPS ground state (DMRG <-> disentangle <-> U H Ud) via
#             run_camps_gs; a separate driver, not wired into this scan (see note
#             in main()).
CAMPS_MODE        = "off"
N_CAMPS_ITERS     = 3              # "gs" mode only
N_CLIFF_SWEEPS    = 2              # Clifford sweeps per pass
CAMPS_ENTROPY     = "vn"           


CAMPS_CHI_EVAL    = 16
CAMPS_TOP_K       = 6
MPO_CHI_CAP       = 32             
CHI_MAGIC         = 32       

COMPUTE_MAGIC     = True
EXACT_MAX_QUBITS  = 14
SAMPLER           = "perfect"      # "perfect", "metropolis", "none"

# Nonlocal magic from the middle-cut entanglement spectrum (Schmidt reference
# state, Liu & Cui arXiv:2607.10714).  Gives M_2^NL with 4F <= M_2^NL <= M_2.
COMPUTE_NL_MAGIC  = True
# Keep only the largest NL_MAGIC_MAX_RANK Schmidt probabilities when the cut rank
# is large (reference state uses 2*ceil(log2 rank) qubits; the exact FWHT there
# is ~feasible to rank a few hundred).  None = use the full spectrum.  The
# discarded Schmidt weight eta is reported so the truncation error stays bounded.
NL_MAGIC_MAX_RANK = 256

# Gauge-invariant (Donnelly) entanglement entropy at the middle vertical cut,
# resolved by the tuple of the Ly crossing-link SU(2) irreps -- the 2D analogue
# of the 1D boundary_irrep_resolved_EE.  Costs up to (irreps)^Ly sector
# projections, so it is off by default; DONNELLY_MAX_SECTORS caps the work.
COMPUTE_DONNELLY_EE  = True
DONNELLY_MAX_SECTORS = 8192

# Exact diagonalization overlay for small lattices: if the total qubit count is
# <= ED_MAX_QUBITS, the same Hamiltonian is diagonalized exactly (TeNPy
# ExactDiag) and the exact eps, <E^2>, <B^2>, Re<B_p>, S, M_2^NL, M_2 are drawn
# as dashed black reference curves.  Skipped automatically for larger lattices.
COMPUTE_ED        = True
ED_MAX_QUBITS     = 12
ED_MAX_SIZE       = 3.0e8          # max dense Hilbert-space size (elements) for ED

# Independent cross-check of M_2^NL: the reference-state value (arXiv:2607.10714)
# is a CONJECTURE, so optionally verify it against the *definition* -- direct
# minimization of M_2 over local unitaries U_A x U_B (Cao et al. arXiv:2502.06393;
# also 2607.07808, 2608.02745).  The minimizer truncates the cut spectrum to the
# largest NL_VALIDATE_KEEP Schmidt values and compares min vs the reference value
# at that same rank.  Works at any lattice size (needs only the top Schmidt
# values), but costs ~a unitary optimization on 2*ceil(log2 keep) qubits.
VALIDATE_NL          = True
NL_VALIDATE_KEEP     = 8
NL_VALIDATE_RESTARTS = 3
# Cross-check optimizer: "lbfgs" = SciPy L-BFGS on exp(iH) parameters (default,
# no extra deps); "riemannian" = Riemannian optimization on the complex unitary
# manifold via pymanopt (the method used in arXiv:2607.10714), which scales
# better to large NL_VALIDATE_KEEP so you can validate at the reported rank.
NL_VALIDATE_METHOD   = "lbfgs"

MC_SAMPLES, MC_BURN, MC_SEEDS = 60000, 6000, (0, 1, 2)   # SAMPLER='metropolis' only

PERFECT_SAMPLES   = 2000
PERFECT_SEQUENTIAL = False         # reference only; production uses batched full sampler
PERFECT_BATCH      = 4             # walkers vectorized inside one NumPy batch
PERFECT_WORKERS    = 6             # independent vectorized batches in parallel
PARALLEL_CHUNK_SAMPLES = 100       # progress/task granularity per worker chunk
RAO_BLACKWELL_LAST = True          # analytically average the final-site Pauli draw

# Exact-to-floating-point factorization of doubled right environments.  Eigenvalues
# below the standard numerical-rank threshold (~dimension*eps) are treated as
# floating-point zeros.  A factor is used only if it actually reduces dimension.
ENV_FACTORIZE      = False
ENV_FACTOR_MIN_SAVING = 0.90       # use Y only when rank <= 90% of full dimension
ENV_RANK_EPS_MULT  = 10.0          # multiplier on dim*machine-epsilon threshold
ENV_RANK_REPORT_RTOLS = (1e-14, 1e-12, 1e-10)

ADAPTIVE_PERFECT   = False         # optional precision-based early stopping
M2_TARGET_ERR      = 0.02
MIN_PERFECT_SAMPLES= 400
CHECK_EVERY        = 100
PROFILE_TIMINGS    = True
WARM_START_G2      = True          # only used when a checkpoint is absent

# Warm-start CONTINUATION in g^2 (upward in g^2).  G2_VALUES are processed in
# ascending order.  For every point that needs a DMRG run the seed is chosen
# automatically (see _warm_start_seed):
#   * g^2 <= COLD_START_G2            -> fresh product-state start, NO seed search;
#   * otherwise the seed is the checkpoint (same LATTICE, MASS and CHI in CKPT_DIR)
#     with the nearest SMALLER g^2 -- including the point just computed earlier in this
#     run, which is taken straight from memory (so chaining works even with
#     SAVE_GS=False);
#   * if no smaller-g^2 checkpoint exists at all -> fresh start.
# E.g. G2_VALUES=[0.2, 0.4, 0.6]: 0.2 fresh, 0.4 <- 0.2, 0.6 <- 0.4.
# G2_VALUES=[0.8] with gs_*_g0.6_* on disk: 0.8 <- 0.6.
# Keeping the mixer on while re-relaxing is what removes the metastable "jag"
# near the transition.  FORCE_RECOMPUTE re-runs the G2_VALUES points even if
# (stale) checkpoints for them already exist; seeds are strictly smaller g^2, and
# with ascending order every smaller point of this run is recomputed first.
COLD_START_G2         = 0.20        # g^2 <= this always starts fresh; None -> rely on auto rule only
WARM_START_SEED_PICK  = "nearest"   # "nearest": largest available g^2 below target
                                    # "smallest": smallest available g^2 below target
WARM_START_SEED_G2    = None        # optional manual override: force this checkpoint's
                                    # g^2 as the seed for the FIRST computed point
                                    # (old behaviour); None -> automatic search
WARM_START_MIXER      = True        # use the DMRG mixer on warm-started runs
WARM_START_MIN_SWEEPS = 4           # min sweeps for a warm-started run
FORCE_RECOMPUTE       = False       # True: recompute G2_VALUES even if their checkpoints exist

# Batch / restart control (used by su2_scan.sh).
#   RESUME   : skip every g^2 that already has a row in the output JSON; a g^2 whose
#              ground-state checkpoint exists but has no JSON row (job killed during
#              the measurements) is re-measured from the checkpoint, no DMRG.
#   DEADLINE : unix time by which this process must be done (job start + walltime
#              - margin).  Before each new g^2 the expected cost (largest per-point
#              walltime seen so far for this lattice/chi, else POINT_HOURS_GUESS) is
#              compared with the time left; if it does not fit the run stops cleanly
#              with exit code EXIT_RESUBMIT so the job script resubmits.
RESUME            = True
DEADLINE          = None            # --deadline <unix time>; None = no limit
POINT_HOURS_GUESS = None            # --point-hours-guess; used before any point is timed
EXIT_RESUBMIT     = 75              # exit code: unfinished, resubmit me
EXIT_SEED_WAIT    = 76              # exit code: --seed-chi state not there yet, retry later
# A DMRG run is checkpointed after EVERY sweep (gs_..._chi<CHI>.partial.pkl: MPS,
# sweep count, mixer amplitude/remaining sweeps), so one g^2 point can span several
# jobs: the next job continues the same DMRG run (same sweep schedule) instead of
# starting over.  The partial file is deleted once the point is converged.

# Bond-dimension ladder for segmented runs (--seed-chi C, used for 8x8 chi=128):
# the FIRST g^2 of the --g2 list starts from the converged chi=C ground state at the
# SAME lattice, mass and g^2 (which itself came up the g^2 ladder from the product
# state at g^2=0.2); every further g^2 of the list starts from the previous point
# of the list at chi=CHI, as usual.  Independent g^2 segments can then run at the
# same time.  None = plain continuation (product state at g^2=0.2).
SEED_CHI          = None
MAX_E_ERR         = 1e-12           # DMRG max_E_err (unchanged default; --max-E-err)

MEM_LIMIT_GB      = 2.0            
VALIDATE_SAMPLER  = True           

SHOW_CHARGE_RESOLVED = False
CHI_BENCH         = []

# ground-state checkpointing (skip DMRG if a matching checkpoint exists)
SAVE_GS           = True
LOAD_GS           = True
CKPT_DIR          = "gs_checkpoints"


LATTICES = {"2x2": (2,2), "3x2": (3,2), "2x3": (2,3), "4x2": (4,2),
            "4x3": (4,3), "4x4": (4,4)}

def parse_lattice(label):
    """Return (Lx, Ly) for any label of the form 'LxxLy' (e.g. '6x6').  The
    LATTICES dict above is only a convenience alias table -- any size works."""
    if label in LATTICES:
        return LATTICES[label]
    try:
        a, b = label.lower().split("x")
        Lx, Ly = int(a), int(b)
    except Exception:
        raise ValueError(f"cannot parse lattice label {label!r}; use e.g. '6x6'")
    if Lx < 1 or Ly < 1:
        raise ValueError(f"lattice dimensions must be >=1, got {Lx}x{Ly}")
    if (Lx * Ly) % 2 != 0:
        log(f"  [warning] |Lambda|={Lx*Ly} is odd; the staggered b=0 sector "
            f"(N_tot=|Lambda|) is only well defined for even |Lambda|.")
    return Lx, Ly
T_HOP, LAM_LINK, LAM_PAD = 0.5, 100.0, 1.0e4
def log(*a): print(*a, flush=True)


Psi_r = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,-1],[0,0,0,0]], dtype=complex)
Psi_g = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=complex)
P_psi = np.diag([1,-1,-1,1]).astype(complex)
N_total = Psi_r.conj().T @ Psi_r + Psi_g.conj().T @ Psi_g
cf = 2 ** (-0.25)   # 2^(-1/4) per rishon -> U^{ab} matrix elements 1/sqrt2 (Cataldi (B33), Peter-Weyl)
Zi = {'r': cf*np.array([[0,1,0],[0,0,0],[1,0,0]], dtype=complex),
      'g': cf*np.array([[0,0,1],[-1,0,0],[0,0,0]], dtype=complex)}
Zi_dag = {c: Zi[c].conj().T for c in 'rg'}
P_leg = np.diag([1,-1,-1]).astype(complex)
Sx = 0.5*np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]], dtype=complex)
Sy = 0.5*np.array([[0,0,0,0],[0,0,-1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
Sz = np.diag([0,0.5,-0.5,0]).astype(complex)
Tx = 0.5*np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex)
Ty = 0.5*np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)
Tz = np.diag([0,0.5,-0.5]).astype(complex)
T2 = np.diag([0,0.75,0.75]).astype(complex)
n_leg = np.diag([0,1,1]).astype(complex)
SLOTS = ['matter','mx','my','px','py']
SLOT_DIM = {'matter':4,'mx':3,'my':3,'px':3,'py':3}
LEGS = ['mx','my','px','py']

def embed(d):
    return reduce(np.kron, [d.get(s, np.eye(SLOT_DIM[s], dtype=complex)) for s in SLOTS])

N_raw   = embed({'matter': N_total})
gamma   = 0.5*sum(embed({l: T2}) for l in LEGS)          # electric E^2
T2_leg  = {l: embed({l: T2}) for l in LEGS}
leg_occ = {l: embed({l: n_leg}) for l in LEGS}
Gx = embed({'matter':Sx}) + sum(embed({l:Tx}) for l in LEGS)
Gy = embed({'matter':Sy}) + sum(embed({l:Ty}) for l in LEGS)
Gz = embed({'matter':Sz}) + sum(embed({l:Tz}) for l in LEGS)
G2_raw = Gx@Gx + Gy@Gy + Gz@Gz                            # Gauss law generator
Z_P     = {c: Zi[c] @ P_leg for c in 'rg'}          # (zeta P)
Zd_P    = {c: Zi_dag[c] @ P_leg for c in 'rg'}      # (zeta^+ P)
P_Z     = {c: P_leg @ Zi[c] for c in 'rg'}          # (P zeta)
# Corner operators = site-by-site regrouping of Tr(U_1 U_2 U_3^+ U_4^+), see App. A.
# Rule for an ORDERED product A_{s1} B_{s2} of two odd on-site operators with the slot ordering
# [matter, mx, my, px, py]:   s1 before s2  ->  (A P)_{s1} P...P (B)_{s2}
#                             s1 after  s2  ->  (P B)_{s2} P...P (A)_{s1}
# corner j        : sum_c zeta_c(px) zeta_c^+(py)
C_pp = sum(embed({'px':Z_P[c],'py':Zi_dag[c]}) for c in 'rg')
# corner j+x      : sum_c zeta_c^+(mx) zeta_c(py)     (mx, py non-adjacent -> P on my, px)
C_mp = sum(embed({'mx':Zd_P[c],'my':P_leg,'px':P_leg,'py':Zi[c]}) for c in 'rg')
# corner j+y      : sum_c zeta_c^+(px) zeta_c(my)     (px after my)
C_pm = sum(embed({'my':P_Z[c],'px':Zi_dag[c]}) for c in 'rg')
# corner j+x+y    : sum_c zeta_c^+(my) zeta_c(mx)     (my after mx)
C_mm = sum(embed({'mx':P_Z[c],'my':Zi_dag[c]}) for c in 'rg')
# With these definitions  Tr box = C_pp(j) C_mp(j+x) C_pm(j+y) C_mm(j+x+y)  exactly; the magnetic term
# -1/(2 g^2) (Tr box + h.c.) then has the Kogut-Susskind sign.
psi_dP = {'r': Psi_r.conj().T @ P_psi, 'g': Psi_g.conj().T @ P_psi}
Qd_raw = {
 'mx': sum(embed({'matter':psi_dP[c],'mx':Zi[c]}) for c in 'rg'),
 'my': sum(embed({'matter':psi_dP[c],'mx':P_leg,'my':Zi[c]}) for c in 'rg'),
 'px': sum(embed({'matter':psi_dP[c],'mx':P_leg,'my':P_leg,'px':Zi[c]}) for c in 'rg'),
 'py': sum(embed({'matter':psi_dP[c],'mx':P_leg,'my':P_leg,'px':P_leg,'py':Zi[c]}) for c in 'rg')}
Q_raw = {l: Qd_raw[l].conj().T for l in LEGS}
RAW_OPS = {'Qd_px':Qd_raw['px'],'Q_px':Q_raw['px'],'Qd_mx':Qd_raw['mx'],'Q_mx':Q_raw['mx'],
           'Qd_py':Qd_raw['py'],'Q_py':Q_raw['py'],'Qd_my':Qd_raw['my'],'Q_my':Q_raw['my'],
           'C_pp':C_pp,'C_mp':C_mp,'C_pm':C_pm,'C_mm':C_mm}
HC = {'Qd_px':'Q_px','Q_px':'Qd_px','Qd_mx':'Q_mx','Q_mx':'Qd_mx',
      'Qd_py':'Q_py','Q_py':'Qd_py','Qd_my':'Q_my','Q_my':'Qd_my'}

#324 -> 6 / 13 / 30 
_basis_cache = {}
def site_basis(frozen, seed=7):
    """eigh alone leaves the degenerate G^2 kernel in an ARBITRARY basis (T^2
       then non-diagonal on edge/bulk sites, and magic is basis-dependent).
       Simultaneously diagonalize G^2, the four T2_leg and N."""
    key = frozenset(frozen)
    if key in _basis_cache: return _basis_cache[key]
    keep = np.ones(324, dtype=bool)
    for leg in key: keep &= (np.abs(np.diag(leg_occ[leg])) < 0.5)
    idx = np.where(keep)[0]
    w, v = np.linalg.eigh(G2_raw[np.ix_(idx, idx)])
    kern = v[:, np.abs(w) < 1e-8]
    M = np.zeros((324, kern.shape[1]), dtype=complex); M[idx,:] = kern
    ops = [T2_leg[l] for l in LEGS] + [N_raw]
    c = np.random.default_rng(seed).random(len(ops)) + 1.0
    A = sum(ci*(M.conj().T@O@M) for ci,O in zip(c,ops)); A = 0.5*(A+A.conj().T)
    M = M @ np.linalg.eigh(A)[1]
    for O in ops:
        r = M.conj().T@O@M
        assert np.abs(r-np.diag(np.diag(r))).max() < 1e-8
    _basis_cache[key] = M
    return M

def frozen_legs_of(Lx, Ly, jx, jy):
    f = []
    if jx == 0:    f.append('mx')
    if jx == Lx-1: f.append('px')
    if jy == 0:    f.append('my')
    if jy == Ly-1: f.append('py')
    return f

# Hamiltonian
_SITE_CACHE = {}
_OBS_MPO_CACHE = {}

def build_sites(Lx, Ly, verbose=False):
    key = (int(Lx), int(Ly))
    if key in _SITE_CACHE:
        sites, qps, coords, Nch, dims = _SITE_CACHE[key]
        if verbose:
            log(f"    dims {dims} -> {[1<<q for q in qps]}, {sum(qps)} qubits [cached]")
        return sites, list(qps), list(coords), [x.copy() for x in Nch]

    coords = [(jx,jy) for jx in range(Lx) for jy in range(Ly)]
    chinfo = ChargeInfo([1], ['N'])
    Npar = N_raw + sum(leg_occ.values())     # TOTAL parity -> all ops EVEN,
    sites, qps, dims, Nch = [], [], [], []   # so no inter-site JW strings
    for (jx,jy) in coords:
        M = site_basis(frozen_legs_of(Lx,Ly,jx,jy)); d = M.shape[1]
        D = 1 << int(np.ceil(np.log2(d))); dims.append(d); qps.append(int(np.log2(D)))
        def red(raw):
            r = M.conj().T@raw@M; r[np.abs(r)<1e-12] = 0
            out = np.zeros((D,D), dtype=complex); out[:d,:d] = r; return out
        par = (-1.0)**np.round(np.real(np.diag(red(Npar)))).astype(int); par[d:] = 1.0
        jw = np.diag(par)
        Nv = np.round(np.real(np.diag(red(N_raw)))).astype(int); Nv[d:] = 0
        Nch.append(Nv)
        s = Site(LegCharge.from_qflat(chinfo, [[int(x)] for x in Nv]), JW=jw)
        s.add_op('P_un', np.diag([0.0]*d + [1.0]*(D-d)))
        Nop = red(N_raw); s.add_op('N', Nop); s.add_op('Nsq', Nop@Nop)
        s.add_op('Gamma', red(gamma))
        for name, raw in RAW_OPS.items():
            op = red(raw)
            assert np.allclose(jw@op@jw, op), f"{name} not even"
            if name.startswith('C'):
                s.add_op(name, op, need_JW=False, hc=name+'d')
                s.add_op(name+'d', op.conj().T, need_JW=False, hc=name)
            else:
                s.add_op(name, op, need_JW=False, hc=HC[name])
        for lg in LEGS:
            t2 = red(T2_leg[lg]); s.add_op('T2_'+lg, t2); s.add_op('T2sq_'+lg, t2@t2)
        sites.append(s)
    if verbose:
        log(f"    dims {dims} -> {[1<<q for q in qps]}, {sum(qps)} qubits")
    _SITE_CACHE[key] = (sites, tuple(qps), tuple(coords), [x.copy() for x in Nch], tuple(dims))
    return sites, qps, coords, Nch

def build_model(lat_tp, Lx, Ly, g2, mass, with_penalty=True):

    B = -1.0/(2.0*g2); idx = lambda x,y: x*Ly + y
    H = CouplingModel(lat_tp)
    for x in range(Lx):
        for y in range(Ly):
            u = idx(x,y)
            # mass
            H.add_local_term(mass*((-1)**(x+y)), [('N',[0,0,u])])
            # electric
            H.add_local_term(g2/2, [('Gamma',[0,0,u])])
            # hopping, x dir
            if x+1 < Lx:
                H.add_local_term(-1j*T_HOP, [('Qd_px',[0,0,u]),
                                             ('Q_mx',[0,0,idx(x+1,y)])], plus_hc=True)
            # hopping, y dir
            if y+1 < Ly:
                H.add_local_term(-T_HOP*((-1)**(x+y)), [('Qd_py',[0,0,u]),
                                             ('Q_my',[0,0,idx(x,y+1)])], plus_hc=True)
            # plaquette
            if x+1 < Lx and y+1 < Ly:
                H.add_local_term(B, [('C_pp',[0,0,u]),('C_mp',[0,0,idx(x+1,y)]),
                                     ('C_pm',[0,0,idx(x,y+1)]),
                                     ('C_mm',[0,0,idx(x+1,y+1)])], plus_hc=True)
            # penalties: padding leakage + link flux matching
            if with_penalty:
                H.add_local_term(LAM_PAD, [('P_un',[0,0,u])])
                for (dx,dy),lh,lt in [((1,0),'px','mx'),((0,1),'py','my')]:
                    nx, ny = x+dx, y+dy
                    if nx >= Lx or ny >= Ly: continue
                    v = idx(nx,ny)
                    H.add_local_term(LAM_LINK,    [('T2sq_'+lh,[0,0,u])])
                    H.add_local_term(LAM_LINK,    [('T2sq_'+lt,[0,0,v])])
                    H.add_local_term(-2*LAM_LINK, [('T2_'+lh,[0,0,u]),
                                                   ('T2_'+lt,[0,0,v])])
    return MPOModel(lat_tp, H.calc_H_MPO())



def build_penalty_mpo(lat_tp, Lx, Ly):
    """Only the gauge/padding penalty MPO.  This is much cheaper than rebuilding
    the full physical Hamiltonian a second time just to measure eps."""
    idx = lambda x,y: x*Ly + y
    H = CouplingModel(lat_tp)
    for x in range(Lx):
        for y in range(Ly):
            u = idx(x,y)
            H.add_local_term(LAM_PAD, [('P_un',[0,0,u])])
            for (dx,dy),lh,lt in [((1,0),'px','mx'),((0,1),'py','my')]:
                nx, ny = x+dx, y+dy
                if nx >= Lx or ny >= Ly: continue
                v = idx(nx,ny)
                H.add_local_term(LAM_LINK,    [('T2sq_'+lh,[0,0,u])])
                H.add_local_term(LAM_LINK,    [('T2sq_'+lt,[0,0,v])])
                H.add_local_term(-2*LAM_LINK, [('T2_'+lh,[0,0,u]),
                                               ('T2_'+lt,[0,0,v])])
    return H.calc_H_MPO()

# ---- gauge observables:  electric <E^2>  and  magnetic <B^2> / plaquette -----
def build_electric_mpo(lat_tp, Lx, Ly):
    """MPO for  sum_j Gamma_j  (Gamma_j = 1/2 sum_mu T^2_{j,mu} = electric E^2)."""
    idx = lambda x, y: x*Ly + y
    H = CouplingModel(lat_tp)
    for x in range(Lx):
        for y in range(Ly):
            H.add_local_term(1.0, [('Gamma', [0, 0, idx(x, y)])])
    return H.calc_H_MPO()

def build_magnetic_mpo(lat_tp, Lx, Ly):
    """MPO for  sum_plaq (B_p + B_p^dag) = 2 sum_plaq Re B_p, with the plaquette
       B_p = Tr box = C_pp(j) C_mp(j+x) C_pm(j+y) C_mm(j+x+y)  (same operator as in the Hamiltonian)."""
    idx = lambda x, y: x*Ly + y
    H = CouplingModel(lat_tp)
    for x in range(Lx-1):
        for y in range(Ly-1):
            H.add_local_term(1.0, [('C_pp', [0, 0, idx(x, y)]),
                                   ('C_mp', [0, 0, idx(x+1, y)]),
                                   ('C_pm', [0, 0, idx(x, y+1)]),
                                   ('C_mm', [0, 0, idx(x+1, y+1)])], plus_hc=True)
    return H.calc_H_MPO()

def _observable_mpos(lat_tp, Lx, Ly):
    # The key must include the Site objects: a checkpoint loaded from disk carries the
    # (pickled) sites of the process that wrote it, whose operators can differ from
    # this process's by degenerate-subspace rotations/phases (see run_plain_gs).  A
    # geometry-only key reused the first MPOs built in a process for all later states.
    key = (int(Lx), int(Ly), tuple(id(s) for s in lat_tp.mps_sites()))
    if key not in _OBS_MPO_CACHE:
        _OBS_MPO_CACHE[key] = (build_electric_mpo(lat_tp, Lx, Ly),
                               build_magnetic_mpo(lat_tp, Lx, Ly))
    return _OBS_MPO_CACHE[key]

def measure_gauge_observables(psi, lat_tp, Lx, Ly):
    """Gauge observables; geometry-only MPOs are cached and reused.
       <E^2>   = (1/N_links) sum_links <E^2_l>   (sum_j Gamma_j = sum_links E^2_l; E^2 = 3/4 on an active link)
       Re<B_p> = (1/N_plaq) sum_plaq Re<Tr box>  (Tr box = C_pp C_mp C_pm C_mm; equals 1 for a fully ordered plaquette)
       <B^2>   = (1 - Re<B_p>)/2."""
    nlinks = Lx*(Ly-1) + Ly*(Lx-1)
    nplaq = max((Lx-1)*(Ly-1), 0)
    He, Hb = _observable_mpos(lat_tp, Lx, Ly)
    E2 = float(np.real(He.expectation_value(psi))) / max(nlinks, 1)
    out = {'E2': E2}
    if nplaq > 0:
        Bsum = float(np.real(Hb.expectation_value(psi)))
        ReBp = Bsum / (2.0*nplaq)
        out['ReBp'] = ReBp
        out['B2'] = 0.5*(1.0 - ReBp)
    else:
        out['ReBp'] = float('nan'); out['B2'] = float('nan')
    return out


I2   = np.eye(2, dtype=complex)
Hg   = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
Sg   = np.diag([1,1j]).astype(complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
SWAPg= np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

_REPS = None
def clifford_coset_reps():
    global _REPS
    if _REPS is not None: return _REPS
    gens = [np.kron(Hg,I2), np.kron(I2,Hg), np.kron(Sg,I2), np.kron(I2,Sg),
            CNOT, SWAPg@CNOT@SWAPg]
    def key(U):
        V = U.copy(); ph = V.flat[int(np.argmax(np.abs(V)>1e-9))]
        return np.round(V*(np.abs(ph)/ph), 6).tobytes()
    seen = {key(np.eye(4,dtype=complex)): np.eye(4,dtype=complex)}
    frontier = [np.eye(4,dtype=complex)]
    for _ in range(9):
        new = []
        for U in frontier:
            for g in gens:
                V = g@U; kk = key(V)
                if kk not in seen: seen[kk] = V; new.append(V)
        frontier = new
        if not frontier: break
    rng = np.random.default_rng(0)
    th = rng.standard_normal((5,4,5)) + 1j*rng.standard_normal((5,4,5))
    th /= np.linalg.norm(th)
    uniq = {}
    for U in seen.values():
        t = np.tensordot(U, th, axes=([1],[1]))
        t = np.moveaxis(t,0,1).reshape(10,10)
        uniq.setdefault(np.round(np.linalg.svd(t, compute_uv=False), 8).tobytes(), U)
    _REPS = list(uniq.values()); return _REPS

def embed_2site_gate(U4, qL, qR, iL, jR):
    n = qL + qR; p1, p2 = iL, qL + jR; D = 1 << n
    Mt = np.eye(D, dtype=complex).reshape([2]*n + [D])
    G = U4.reshape(2,2,2,2)
    Mt = np.tensordot(G, Mt, axes=([2,3],[p1,p2]))
    rest = [ax for ax in range(n) if ax not in (p1,p2)]
    pos = [0]*n; pos[p1], pos[p2] = 0, 1
    for i, ax in enumerate(rest): pos[ax] = i + 2
    Mt = np.transpose(Mt, [pos[j] for j in range(n)] + [n])
    return Mt.reshape(D, D)

# MPO
def apply_gate_theta(theta, U4, qL, qR, iL, jR):
    vL, d1, d2, vR = theta.shape
    n = qL + qR
    a1, a2 = 1 + iL, 1 + qL + jR
    t = theta.reshape([vL] + [2]*n + [vR])
    t = np.tensordot(U4.reshape(2,2,2,2), t, axes=([2,3],[a1,a2]))
    rest = [ax for ax in range(n+2) if ax not in (a1, a2)]
    perm = []
    for ax in range(n+2):
        if   ax == a1: perm.append(0)
        elif ax == a2: perm.append(1)
        else:          perm.append(2 + rest.index(ax))
    return np.transpose(t, perm).reshape(vL, d1, d2, vR)


_GATE_CACHE = {}
def gate_cached(gi, qL, qR, iL, jR):
    key = (gi, qL, qR, iL, jR)
    U = _GATE_CACHE.get(key)
    if U is None:
        U = embed_2site_gate(clifford_coset_reps()[gi], qL, qR, iL, jR)
        _GATE_CACHE[key] = U
    return U


def _op_schmidt_2q(U4, tol=1e-12):
    Mt = U4.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)   # (a1,b1),(a2,b2)
    W, s, Vh = np.linalg.svd(Mt)
    k = s > tol*s[0]
    us = [W[:, m].reshape(2,2)*s[m] for m in range(4) if k[m]]
    vs = [Vh[m, :].reshape(2,2)      for m in range(4) if k[m]]
    return us, vs

def _embed_1q(u, q, i):
    mats = [np.eye(2, dtype=complex)]*q
    mats[i] = u
    return reduce(np.kron, mats)

def conjugate_mpo_fast(Ws, k, U4, qL, qR, iL, jR, tol=1e-11, chi_cap=None):

    A, B = Ws[k], Ws[k+1]
    DL, Dm, d1, _ = A.shape
    _,  DR, d2, _ = B.shape
    us, vs = _op_schmidt_2q(U4)
    r = len(us)
    Um = [_embed_1q(u, qL, iL) for u in us]
    Vm = [_embed_1q(v, qR, jR) for v in vs]

    A2 = np.empty((DL, Dm*r*r, d1, d1), dtype=complex)
    B2 = np.empty((Dm*r*r, DR, d2, d2), dtype=complex)
    for m in range(r):
        for nn in range(r):
            sl = slice((m*r+nn)*Dm, (m*r+nn+1)*Dm)
            A2[:, sl] = np.einsum('pP,abPQ,Qq->abpq', Um[m], A,
                                  Um[nn].conj().T, optimize=True)
            B2[sl]    = np.einsum('pP,abPQ,Qq->abpq', Vm[m], B,
                                  Vm[nn].conj().T, optimize=True)
    # recompress the widened bond (small SVD)
    mat = A2.transpose(0,2,3,1).reshape(DL*d1*d1, Dm*r*r)
    U_, s, Vh = np.linalg.svd(mat, full_matrices=False)
    keep = s > tol*max(s[0], 1e-300)
    if chi_cap:
        keep[chi_cap:] = False
    U_, s, Vh = U_[:, keep], s[keep], Vh[keep, :]
    rk = len(s)
    out = list(Ws)
    out[k]   = U_.reshape(DL, d1, d1, rk).transpose(0, 3, 1, 2)
    out[k+1] = np.tensordot(np.diag(s) @ Vh, B2, axes=([1],[0]))
    return out


def to_chargefree_mps(psi, fsites, perm):
    Bs = []
    for k in range(psi.L):
        B = psi.get_B(k, form='B').to_ndarray()          # (vL, p, vR)
        p = np.asarray(perm[k])
        Bo = np.zeros_like(B); Bo[:, p, :] = B
        Bs.append(npc.Array.from_ndarray_trivial(Bo, labels=['vL','p','vR']))
    SVs = [np.asarray(psi.get_SL(k)) for k in range(psi.L)] + [np.ones(1)]
    out = MPS(fsites, Bs, SVs, bc='finite', form='B')
    out.canonical_form()
    return out



def conjugate_mpo(Ws, k, Ufull, tol=1e-12, chi_max=None):

    A, B = Ws[k], Ws[k+1]
    DL, Dm, d1, _ = A.shape
    _,  DR, d2, _ = B.shape
    T = np.tensordot(A, B, axes=([1],[0]))            # (wL,p1,p1*,wR,p2,p2*)
    T = np.transpose(T, (0,1,4,2,5,3))                # (wL,p1,p2,p1*,p2*,wR)
    T = T.reshape(DL, d1*d2, d1*d2, DR)
    T = np.einsum('PR,aRSb,SQ->aPQb', Ufull, T, Ufull.conj().T, optimize=True)
    T = T.reshape(DL, d1, d2, d1, d2, DR)
    T = np.transpose(T, (0,1,3,2,4,5))                # (wL,p1,p1*,p2,p2*,wR)
    U_, s, Vh = np.linalg.svd(T.reshape(DL*d1*d1, d2*d2*DR), full_matrices=False)
    keep = s > tol*max(s[0], 1e-300)
    if chi_max: keep[chi_max:] = False
    U_, s, Vh = U_[:,keep], s[keep], Vh[keep,:]
    r = len(s)
    out = list(Ws)
    out[k]   = U_.reshape(DL, d1, d1, r).transpose(0,3,1,2)
    out[k+1] = (np.diag(s)@Vh).reshape(r, d2, d2, DR).transpose(0,3,1,2)
    return out

def build_mpo(fsites, Ws):
    ws = [npc.Array.from_ndarray_trivial(W, labels=['wL','wR','p','p*']) for W in Ws]
    L = len(Ws)
    return MPO(fsites, ws, bc='finite', IdL=[0]+[None]*L, IdR=[None]*L+[0])

def unsort_W(W, perm):
    p = np.asarray(perm)
    t = np.zeros_like(W); t[:, :, p, :] = W
    o = np.zeros_like(W); o[:, :, :, p] = t
    return o

def trim_boundaries(Ws, IdL0, IdRL):
    Ws = [W.copy() for W in Ws]
    Ws[0] = Ws[0][IdL0:IdL0+1]; Ws[-1] = Ws[-1][:, IdRL:IdRL+1]
    return Ws


def _S(sv, kind=None):
    kind = CAMPS_ENTROPY if kind is None else kind
    p = np.asarray(sv)**2; z = p.sum()
    if z <= 0: return 0.0
    p = p/z; p = p[p > 1e-16]
    if kind == "vn": return float(-np.sum(p*np.log2(p)))
    return float(-np.log2(np.sum(p**2)))

def camps_disentangle(psi, qps, n_sweeps=2, verbose=False):

    group = clifford_coset_reps(); L = psi.L; circuit = []
    S0 = [float(x)/np.log(2) for x in psi.entanglement_entropy(n=1)]
    for sw in range(n_sweeps):
        for k in range(L-1):
            qL, qR = qps[k], qps[k+1]
            theta = psi.get_theta(k, 2).to_ndarray()
            vL, d1, d2, vR = theta.shape
            flat = theta.reshape(vL, d1*d2, vR)
            best = _S(np.linalg.svd(theta.reshape(vL*d1, d2*vR), compute_uv=False))
            bestU, bestG = None, None
            for iL in range(qL):
                for jR in range(qR):
                    for gi, U4 in enumerate(group):
                        th = apply_gate_theta(theta, U4, qL, qR, iL, jR)
                        val = _S(np.linalg.svd(th.reshape(vL*d1, d2*vR),
                                               compute_uv=False))
                        if val < best - 1e-12:
                            best, bestU, bestG = (val,
                                embed_2site_gate(U4,qL,qR,iL,jR), (k,iL,jR,gi))
            if bestU is not None:
                op = npc.Array.from_ndarray_trivial(
                    bestU.reshape(d1,d2,d1,d2), labels=['p0','p1','p0*','p1*'])
                psi.apply_local_op(k, op, unitary=True, renormalize=True)
                circuit.append(bestG)
        if verbose:
            Sn = [float(x)/np.log(2) for x in psi.entanglement_entropy(n=1)]
            log(f"      cliff sweep {sw+1}: sum S {sum(S0):.4f} -> {sum(Sn):.4f}"
                f"  ({len(circuit)} gates)")
    # apply_local_op leaves O(1e-3) canonical-form error after many gates
    if circuit: psi.canonical_form()
    S1 = [float(x)/np.log(2) for x in psi.entanglement_entropy(n=1)]
    return circuit, {'S_before_tot': float(sum(S0)), 'S_after_tot': float(sum(S1)),
                     'n_gates': len(circuit)}

def run_dmrg_on(Ws, fsites, chi_max, psi=None, max_sweeps=None, E_tol=1e-12):
    max_sweeps = MAX_SWEEPS if max_sweeps is None else max_sweeps
    flat = TeNPyLattice([1,1], fsites, bc='open', bc_MPS='finite')
    model = MPOModel(flat, build_mpo(fsites, Ws))
    psi = MPS.from_product_state(fsites, [0]*len(fsites), bc='finite') \
          if psi is None else psi.copy()
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'trunc_params': {'chi_max': int(chi_max), 'svd_min': 1e-14},
        'max_sweeps': int(max_sweeps), 'min_sweeps': 4, 'mixer': True,
        'max_E_err': E_tol, 'combine': False, 'max_trunc_err': 1.0,
        'max_N_sites_per_ring': 10**9})   # all sites live in one 1x1 unit cell
    E, _ = eng.run()
    return float(np.real(E)), psi

def run_camps_gs(Lx, Ly, g2, chi_max, n_camps_iters=None, n_cliff_sweeps=None,
                 verbose=True):
    n_camps_iters  = N_CAMPS_ITERS  if n_camps_iters  is None else n_camps_iters
    n_cliff_sweeps = N_CLIFF_SWEEPS if n_cliff_sweeps is None else n_cliff_sweeps
    sites, qps, coords, Nch = build_sites(Lx, Ly, verbose=False)
    lat  = TeNPyLattice([1,1], sites, bc='open', bc_MPS='finite')
    H  = build_model(lat, Lx, Ly, g2, MASS, True).H_MPO
    Hp = build_model(lat, Lx, Ly, g2, MASS, False).H_MPO
    L, n = H.L, Lx*Ly
    perm = [np.asarray(getattr(s,'perm', np.arange(s.dim))) for s in sites]
    # rotate the MPO into the PHYSICAL basis before any Clifford touches it
    Ws  = trim_boundaries([unsort_W(H.get_W(i).to_ndarray(),  perm[i])
                           for i in range(L)], H.IdL[0],  H.IdR[L])
    Wsp = trim_boundaries([unsort_W(Hp.get_W(i).to_ndarray(), perm[i])
                           for i in range(L)], Hp.IdL[0], Hp.IdR[L])
    fsites = [Site(LegCharge.from_trivial(s.dim)) for s in sites]

    E, psi = run_dmrg_on(Ws, fsites, chi_max)
    eps0 = float(np.real(build_mpo(fsites, Wsp).expectation_value(psi)))/n
    if verbose:
        log(f"    plain DMRG    eps = {eps0:.10f}  chi={max(psi.chi)}"
            f"  sum S={sum(psi.entanglement_entropy(n=1))/np.log(2):.4f}")
    circuits, mpo_chi = [], [max(W.shape[1] for W in Ws)]
    for it in range(n_camps_iters):
        circ, info = camps_disentangle(psi, qps, n_sweeps=n_cliff_sweeps,
                                       verbose=verbose)
        if not circ:
            if verbose: log("    no Clifford accepted, CAMPS converged")
            break
        for (k, iL, jR, gi) in circ:
            U4 = clifford_coset_reps()[gi]
            Ws  = conjugate_mpo_fast(Ws,  k, U4, qps[k], qps[k+1], iL, jR,
                                     chi_cap=MPO_CHI_CAP)
            Wsp = conjugate_mpo_fast(Wsp, k, U4, qps[k], qps[k+1], iL, jR,
                                     chi_cap=MPO_CHI_CAP)
        circuits.append(circ); mpo_chi.append(max(W.shape[1] for W in Ws))
        E, psi = run_dmrg_on(Ws, fsites, chi_max, psi=psi)
        if verbose:
            eps = float(np.real(build_mpo(fsites,Wsp).expectation_value(psi)))/n
            log(f"    CAMPS iter {it+1}: eps = {eps:.10f}  chi={max(psi.chi)}"
                f"  sum S={sum(psi.entanglement_entropy(n=1))/np.log(2):.4f}"
                f"  MPO chi={mpo_chi[-1]}  ({len(circ)} gates)")
    Hpf = build_mpo(fsites, Wsp)
    eps = float(np.real(Hpf.expectation_value(psi)))/n
    return {'eps': eps, 'eps_plain': eps0, 'psi': psi, 'qps': qps, 'perm': perm,
            'sites': sites, 'chi': int(max(psi.chi)), 'mpo_chi': mpo_chi,
            'n_gates': sum(len(c) for c in circuits),
            'S_tot': float(sum(psi.entanglement_entropy(n=1)))/np.log(2)}

def _diagonalize_mixer_S(psi):
    """A sweep with the DMRG mixer leaves 2D matrices in psi._S.  Recover diagonal
    Schmidt values by SVD -- an exact gauge transformation (same state); this is the
    MPS part of TeNPy's Sweep.mixer_cleanup.  Needed to resume from a checkpoint
    written while the mixer was active."""
    first = 1 if psi.finite else 0
    for i in range(first, psi.L):
        S = psi.get_SL(i)
        if S.ndim == 1:
            continue
        U, S, V = npc.svd(S, full_matrices=False, inner_labels=['vR', 'vL'])
        _, form_L = psi.form[psi._to_valid_site_index(i - 1)]
        form_R, _ = psi.form[i]
        B_L = psi.get_B(i - 1, form=None)
        B_R = psi.get_B(i, form=None)
        if form_L == 0.0:
            B_L = npc.tensordot(B_L, U, ['vR', 'vL'])
        elif form_L == 1.0:
            B_L = npc.tensordot(B_L, V.conj().replace_labels(['vR*', 'vL*'], ['vL', 'vR']),
                                ['vR', 'vL'])
        else:
            raise RuntimeError(f"unexpected MPS form {psi.form[i - 1]} at site {i - 1}")
        if form_R == 0.0:
            B_R = npc.tensordot(V, B_R, ['vR', 'vL'])
        elif form_R == 1.0:
            B_R = npc.tensordot(U.conj().replace_labels(['vR*', 'vL*'], ['vL', 'vR']), B_R,
                                ['vR', 'vL'])
        else:
            raise RuntimeError(f"unexpected MPS form {psi.form[i]} at site {i}")
        psi.set_B(i - 1, B_L, form=psi.form[i - 1])
        psi.set_SL(i, S)
        psi.set_B(i, B_R, form=psi.form[i])

def run_plain_gs(Lx, Ly, g2, chi_max, psi0=None, verbose=True, resume=None, on_sweep=None):
    """resume   : dict saved by the per-sweep hook of an interrupted run of THIS
                  point -> continue that DMRG run (same options; sweep count and
                  mixer schedule restored exactly).
       on_sweep : callable(engine), called by TeNPy after every completed sweep
                  (used to write the .partial.pkl checkpoint and to stop in time)."""
    t_build = time.perf_counter()
    sites, qps, coords, Nch = build_sites(Lx, Ly, verbose=False)
    lat = TeNPyLattice([1,1], sites, bc='open', bc_MPS='finite')
    model = build_model(lat, Lx, Ly, g2, MASS, True)
    Hpen = build_penalty_mpo(lat, Lx, Ly)
    t_build = time.perf_counter() - t_build

    mixer_params = None
    if resume is not None:
        psi = resume['data']['psi']
        _diagonalize_mixer_S(psi)         # saved while the mixer was on -> 2D S matrices
        use_mixer = resume['mixer'] is not None
        if use_mixer:
            mixer_params = dict(resume['mixer'])   # amplitude, decay, disable_after
        # TeNPy checks convergence before the first sweep of a resumed run (empty
        # sweep statistics -> IndexError); requiring one more sweep avoids that.
        # The max_sweeps stop (the usual one at max_E_err=1e-12) is unchanged.
        min_sweeps = max(resume['min_sweeps'], resume['sweeps'] + 1)
    elif psi0 is None:
        prod = [int(np.where(Nch[k] == (0 if (c[0]+c[1]) % 2 == 0 else 2))[0][0])
                for k, c in enumerate(coords)]
        psi = MPS.from_product_state(lat.mps_sites(), prod, bc='finite')
        use_mixer = True
        min_sweeps = 4
    else:
        # Continuation in g^2: same lattice/sites/charge structure, nearby GS.
        # Keep the mixer on and take a few min sweeps so the warm-started state
        # can relax out of any metastable basin near the transition.
        psi = psi0.copy()
        use_mixer = WARM_START_MIXER
        min_sweeps = WARM_START_MIN_SWEEPS

    # The DMRG below optimizes psi in THIS process's site basis (the operators of
    # `sites`).  site_basis() comes from eigh, whose choice inside degenerate pairs
    # (6 on bulk, 1 on edge sites) and whose phases depend on the LAPACK build, so a
    # seed / resume MPS pickled by another machine carries foreign Site objects.  Keep
    # them and every later measurement from psi.sites (checkpoint reload, cached
    # observable MPOs) uses operators in the wrong basis: Re<B_p> goes wrong while
    # eps, <E^2> (diagonal) and the Schmidt spectrum stay correct.  Relabel first.
    psi.sites = list(lat.mps_sites())

    opts = {
        'trunc_params': {'chi_max': int(chi_max), 'svd_min': 1e-14},
        'max_sweeps': MAX_SWEEPS, 'min_sweeps': min_sweeps, 'mixer': use_mixer,
        'max_E_err': MAX_E_ERR, 'combine': False, 'max_trunc_err': 1.0,
        'max_N_sites_per_ring': 10**9}    # all sites live in one 1x1 unit cell
    if mixer_params is not None:
        opts['mixer_params'] = mixer_params
    eng = dmrg.TwoSiteDMRGEngine(psi, model, opts,
                                 resume_data=None if resume is None else resume['data'])
    if on_sweep is not None:
        eng.checkpoint.connect(on_sweep)
    t_dmrg = time.perf_counter()
    Etotal, _ = eng.run()
    psi = eng.psi
    t_dmrg = time.perf_counter() - t_dmrg

    n = Lx*Ly
    # H_total = H_phys + H_penalty.  Avoid constructing H_phys separately.
    Epen = float(np.real(Hpen.expectation_value(psi)))
    eps = (float(np.real(Etotal)) - Epen)/n
    t_obs = time.perf_counter()
    obs = measure_gauge_observables(psi, lat, Lx, Ly)
    t_obs = time.perf_counter() - t_obs
    if verbose and PROFILE_TIMINGS:
        log(f"    timings: build={t_build:.2f}s  DMRG={t_dmrg:.2f}s  observables={t_obs:.2f}s")
    return {'eps': eps, 'eps_plain': eps, 'psi': psi, 'qps': qps,
            'perm': [np.asarray(getattr(s,'perm',np.arange(s.dim))) for s in sites],
            'sites': sites, 'chi': int(max(psi.chi)), 'mpo_chi': [],
            'n_gates': 0, 'S_tot': float(sum(psi.entanglement_entropy(n=1)))/np.log(2),
            'E2': obs['E2'], 'B2': obs['B2'], 'ReBp': obs['ReBp'],
            'Epen': Epen,
            'timing_build': t_build, 'timing_dmrg': t_dmrg, 'timing_obs': t_obs}

def run_camps_warm(Lx, Ly, g2, chi_max, n_camps_iters=2, n_cliff_sweeps=1,
                   u1_sweeps=40, refine_sweeps=8, verbose=True):

    sites, qps, coords, Nch = build_sites(Lx, Ly, verbose=False)
    lat  = TeNPyLattice([1,1], sites, bc='open', bc_MPS='finite')
    model = build_model(lat, Lx, Ly, g2, MASS, True)
    phys  = build_model(lat, Lx, Ly, g2, MASS, False)
    n, L = Lx*Ly, len(sites)
    P = {'trunc_params': {'chi_max': int(chi_max), 'svd_min': 1e-14},
         'max_N_sites_per_ring': 10**9,   # all sites live in one 1x1 unit cell
         'mixer': True, 'max_E_err': 1e-13, 'combine': False,
         'max_trunc_err': 1.0, 'min_sweeps': 4}


    prod = [int(np.where(Nch[k] == (0 if (c[0]+c[1]) % 2 == 0 else 2))[0][0])
            for k, c in enumerate(coords)]
    psi_u1 = MPS.from_product_state(lat.mps_sites(), prod, bc='finite')
    t0 = time.time()
    dmrg.TwoSiteDMRGEngine(psi_u1, model, dict(P, max_sweeps=u1_sweeps)).run()
    t_u1 = time.time() - t0
    eps0 = float(np.real(phys.H_MPO.expectation_value(psi_u1)))/n
    chi0 = int(max(psi_u1.chi))


    perm = [np.asarray(getattr(s,'perm', np.arange(s.dim))) for s in sites]
    H, Hp = model.H_MPO, phys.H_MPO
    Ws  = trim_boundaries([unsort_W(H.get_W(i).to_ndarray(),  perm[i])
                           for i in range(L)], H.IdL[0],  H.IdR[L])
    Wsp = trim_boundaries([unsort_W(Hp.get_W(i).to_ndarray(), perm[i])
                           for i in range(L)], Hp.IdL[0], Hp.IdR[L])
    fsites = [Site(LegCharge.from_trivial(s.dim)) for s in sites]
    psi = to_chargefree_mps(psi_u1, fsites, perm)
    chk = float(np.real(build_mpo(fsites, Wsp).expectation_value(psi)))/n
    assert abs(chk - eps0) < 1e-8, f"conversion broke the state: {chk} vs {eps0}"


    t1 = time.time(); ngates = 0
    for it in range(n_camps_iters):
        circ, info = camps_disentangle(psi, qps, n_sweeps=n_cliff_sweeps,
                                       verbose=verbose)
        if not circ:
            break
        for (k, iL, jR, gi) in circ:
            U4 = clifford_coset_reps()[gi]
            Ws  = conjugate_mpo_fast(Ws,  k, U4, qps[k], qps[k+1], iL, jR,
                                     chi_cap=MPO_CHI_CAP)
            Wsp = conjugate_mpo_fast(Wsp, k, U4, qps[k], qps[k+1], iL, jR,
                                     chi_cap=MPO_CHI_CAP)
        ngates += len(circ)
        _, psi = run_dmrg_on(Ws, fsites, chi_max, psi=psi,
                             max_sweeps=refine_sweeps)
    t_camps = time.time() - t1
    eps = float(np.real(build_mpo(fsites, Wsp).expectation_value(psi)))/n
    return {'eps': eps, 'eps_u1': eps0, 'chi_u1': chi0,
            'chi': int(max(psi.chi)), 't_u1': t_u1, 't_camps': t_camps,
            't_total': t_u1 + t_camps, 'n_gates': ngates, 'psi': psi,
            'qps': qps, 'S_tot': float(sum(psi.entanglement_entropy(n=1)))/np.log(2)}


class CanonMPSd:
    """Mixed-canonical MPS with VARIABLE local dims (dressed sites 8/16/32).
       A[j] left-canonical for j<c, right-canonical for j>c, A[c] holds norm.
       With the centre at c, the SVD of A[c].A[c+1] gives the EXACT Schmidt
       spectrum of the cut between site c and c+1 -- no environment needed."""
    def __init__(self, As, chi_max=None, tol=1e-14):
        self.A = [np.asarray(a, dtype=complex).copy() for a in As]
        self.chi, self.tol = chi_max, tol
        L = len(self.A)
        for k in range(L-1):      self._sr(k)
        for k in range(L-1,0,-1): self._sl(k)
        self.c = 0
        self.A[0] /= np.linalg.norm(self.A[0])
    def _keep(self, sv):
        k = sv > self.tol
        if self.chi: k[self.chi:] = False
        if not k.any(): k[0] = True
        return k
    def _sr(self, k):
        cL,d,cR = self.A[k].shape
        U,s,Vh = np.linalg.svd(self.A[k].reshape(cL*d,cR), full_matrices=False)
        m = self._keep(s); U,s,Vh = U[:,m],s[m],Vh[m,:]
        self.A[k] = U.reshape(cL,d,len(s))
        self.A[k+1] = np.tensordot(np.diag(s)@Vh, self.A[k+1], axes=([1],[0]))
    def _sl(self, k):
        cL,d,cR = self.A[k].shape
        U,s,Vh = np.linalg.svd(self.A[k].reshape(cL,d*cR), full_matrices=False)
        m = self._keep(s); U,s,Vh = U[:,m],s[m],Vh[m,:]
        self.A[k] = Vh.reshape(len(s),d,cR)
        self.A[k-1] = np.tensordot(self.A[k-1], U@np.diag(s), axes=([2],[0]))
    def move_to(self, k):
        while self.c < k: self._sr(self.c); self.c += 1
        while self.c > k: self._sl(self.c); self.c -= 1
    def theta(self):
        k = self.c
        return np.tensordot(self.A[k], self.A[k+1], axes=([2],[0]))  # vL,d1,d2,vR
    def apply(self, U):
        k = self.c; th = self.theta(); vL,d1,d2,vR = th.shape
        th = np.einsum('PQ,aQb->aPb', U, th.reshape(vL,d1*d2,vR), optimize=True)
        Um,s,Vh = np.linalg.svd(th.reshape(vL*d1, d2*vR), full_matrices=False)
        m = self._keep(s); Um,s,Vh = Um[:,m],s[m],Vh[m,:]
        self.A[k]   = (Um@np.diag(s)).reshape(vL,d1,len(s))
        self.A[k+1] = Vh.reshape(len(s),d2,vR)
    def spectrum(self, cut):
        self.move_to(cut-1); th = self.theta()
        vL,d1,d2,vR = th.shape
        s = np.linalg.svd(th.reshape(vL*d1, d2*vR), compute_uv=False)
        return s/np.linalg.norm(s)
    def tensors(self): return [a.copy() for a in self.A]

def camps_reduce(As, qps, n_sweeps=None, chi_eval=None, top_k=None,
                 chi_max=None, verbose=False):

    n_sweeps = N_CLIFF_SWEEPS   if n_sweeps is None else n_sweeps
    chi_eval = CAMPS_CHI_EVAL   if chi_eval is None else chi_eval
    top_k    = CAMPS_TOP_K      if top_k    is None else top_k
    group = clifford_coset_reps()
    psi = CanonMPSd(As, chi_max=chi_max)
    L = len(psi.A)
    cuts = list(range(1, L))
    S0 = [_S(psi.spectrum(c)) for c in cuts]
    chi0 = max(a.shape[2] for a in psi.A)
    ngates = 0
    for sw in range(n_sweeps):
        for k in range(L-1):
            qL, qR = qps[k], qps[k+1]
            psi.move_to(k)
            th = psi.theta(); vL,d1,d2,vR = th.shape
            base = _S(np.linalg.svd(th.reshape(vL*d1, d2*vR), compute_uv=False))
            #cheap screen on a truncated theta
            e1, e2 = min(vL, chi_eval), min(vR, chi_eval)
            ths = th[:e1, :, :, :e2]
            flat_s = ths.reshape(e1, d1*d2, e2)
            cand = []
            for iL in range(qL):
                for jR in range(qR):
                    for gi, U4 in enumerate(group):
                        t = apply_gate_theta(ths, U4, qL, qR, iL, jR)
                        sv = np.linalg.svd(t.reshape(e1*d1, d2*e2), compute_uv=False)
                        cand.append((_S(sv), iL, jR, gi))
            cand.sort(key=lambda x: x[0])
            # exact on the best few 
            flat = th.reshape(vL, d1*d2, vR)
            best, bestU = base, None
            for (_a, iL, jR, gi) in cand[:top_k]:
                t = apply_gate_theta(th, group[gi], qL, qR, iL, jR)
                sv = np.linalg.svd(t.reshape(vL*d1, d2*vR), compute_uv=False)
                val = _S(sv)
                if val < best - 1e-12:
                    best, bestU = val, embed_2site_gate(group[gi],qL,qR,iL,jR)
            if bestU is not None:
                psi.apply(bestU); ngates += 1
        if verbose:
            Sn = [_S(psi.spectrum(c)) for c in cuts]
            log(f"      camps sweep {sw+1}: sum S {sum(S0):.4f} -> {sum(Sn):.4f}"
                f"   chi {chi0} -> {max(a.shape[2] for a in psi.A)}")
    if CHI_MAGIC:
        psi = CanonMPSd(psi.tensors(), chi_max=CHI_MAGIC)
    S1 = [_S(psi.spectrum(c)) for c in cuts]
    psi.move_to(0)
    return psi.tensors(), {'S_before': float(sum(S0)), 'S_after': float(sum(S1)),
                           'chi_before': chi0,
                           'chi_after': max(a.shape[2] for a in psi.A),
                           'n_gates': ngates}


def mps_arrays(psi, perm=None):
  
    As = []
    for k in range(psi.L):
        B = psi.get_B(k, form='B').to_ndarray()
        if perm is not None:
            p = np.asarray(perm[k])
            Bo = np.zeros_like(B); Bo[:, p, :] = B; B = Bo
        As.append(B)
    return As

def split_sites_to_qubits(As, qps, tol=1e-14):
    out = []
    for A, q in zip(As, qps):
        cL, d, cR = A.shape; Mm = A.reshape(cL, *([2]*q), cR); left = cL
        for _ in range(q-1):
            Mm = Mm.reshape(left*2, -1)
            U, sv, Vh = np.linalg.svd(Mm, full_matrices=False)
            k = sv > tol; U, sv, Vh = U[:,k], sv[k], Vh[k,:]
            out.append(U.reshape(left,2,len(sv))); Mm = np.diag(sv)@Vh; left = len(sv)
        out.append(Mm.reshape(left,2,cR))
    return out

def mps_to_vector(As):
    v = As[0]
    for A in As[1:]: v = np.tensordot(v, A, axes=([-1],[0]))
    v = v.reshape(-1); return v/np.linalg.norm(v)

def fwht(a):
    """Unnormalised Walsh-Hadamard transform of the last axis, vectorised over
    all leading axes (batched).  a.shape[-1] must be a power of 2."""
    A = np.asarray(a, dtype=complex).copy()
    lead = A.shape[:-1]
    N = A.shape[-1]
    A = A.reshape(-1, N)
    B = A.shape[0]
    h = 1
    while h < N:
        A = A.reshape(B, N // (2*h), 2, h)
        x = A[:, :, 0, :].copy()
        y = A[:, :, 1, :].copy()
        A[:, :, 0, :] = x + y
        A[:, :, 1, :] = x - y
        A = A.reshape(B, N)
        h *= 2
    return A.reshape(*lead, N)

_POPCACHE = {}
def _popcount_table(N):
    t = _POPCACHE.get(N)
    if t is None:
        t = np.zeros(N, dtype=np.int8)
        for b in range(N.bit_length()-1):
            t[1 << b:1 << (b+1)] = t[:1 << b] + 1
        _POPCACHE[N] = t
    return t

def m2_exact_fwht(v, n, tol=1e-14):
    """Exact M_2 = -log2((1/2^n) sum_P Re<P>^4) via one batched Walsh-Hadamard
    transform per X-flip pattern x.  Batching the x-patterns keeps this fast even
    when the state's support is spread out (e.g. after Clifford disentangling)."""
    N = 1 << n
    supp = np.where(np.abs(v) > tol)[0]
    inS = np.zeros(N, dtype=bool); inS[supp] = True
    xs = np.unique((supp[:, None] ^ supp[None, :]).ravel())
    POP = _popcount_table(N)
    zs = np.arange(N)
    tot = 0.0
    bs = max(1, min(len(xs), 4_000_000 // N))
    vc = np.conj(v)
    for s in range(0, len(xs), bs):
        xb = xs[s:s+bs]
        D = np.zeros((len(xb), N), dtype=complex)
        for r, x in enumerate(xb):
            j = supp ^ x; ok = inS[j]
            if ok.any():
                np.add.at(D[r], supp[ok], vc[supp[ok]] * v[j[ok]])
        C = fwht(D)
        pm = (POP[(xb[:, None] & zs[None, :])] & 3)
        re = np.select([pm == 0, pm == 1, pm == 2],
                       [C.real, -C.imag, -C.real], default=C.imag)
        tot += float(np.sum(re**4))
    return -np.log2(tot/N)


_P1Q = [np.eye(2,dtype=complex), np.array([[0,1],[1,0]],dtype=complex),
        np.array([[0,-1j],[1j,0]],dtype=complex), np.diag([1,-1]).astype(complex)]
def _site_pauli(labels):
    m = _P1Q[labels[0]]
    for k in labels[1:]: m = np.kron(m, _P1Q[k])
    return m




class SparsePauliEval:
    def __init__(self, psi, n, tol=1e-12):
        self.n = n
        supp = np.where(np.abs(psi) > tol)[0].astype(np.int64)
        o = np.argsort(supp)
        self.supp = supp[o]; self.amp = psi[supp][o]
        self.ampc = self.amp.conj(); self.ns = len(self.supp)
    def expval(self, paulis):
        xm = zm = nY = 0
        for q,p in enumerate(paulis):
            bit = 1 << (self.n-1-q)
            if   p == 1: xm |= bit
            elif p == 3: zm |= bit
            elif p == 2: xm |= bit; zm |= bit; nY += 1
        j = self.supp ^ np.int64(xm)
        k = np.searchsorted(self.supp, j); np.clip(k, 0, self.ns-1, out=k)
        ok = self.supp[k] == j
        if not ok.any(): return 0.0
        x = self.supp[ok] & np.int64(zm)
        if hasattr(np, "bitwise_count"):
            # cast to int64 FIRST: bitwise_count returns uint8 and
            # 1-2*(bc&1) would underflow to 255 instead of -1
            par = (np.bitwise_count(x) & 1).astype(np.int64)
        else:
            for sh in (32,16,8,4,2,1): x = x ^ (x >> sh)
            par = (x & 1).astype(np.int64)
        return float(np.real((1j)**nY * np.sum(self.ampc[ok]*(1-2*par)*self.amp[k[ok]])))


class MPSPauliEval:

    def __init__(self, Q, qps):
        self.qps = list(qps); self.n = int(sum(qps))
        self.As, i = [], 0
        for q in self.qps:
            A = Q[i]
            for j in range(1, q): A = np.tensordot(A, Q[i+j], axes=([-1],[0]))
            cL, cR = A.shape[0], A.shape[-1]
            self.As.append(A.reshape(cL, 2**q, cR)); i += q
    def expval(self, paulis):
        E = np.ones((1,1), dtype=complex); off = 0
        for A, q in zip(self.As, self.qps):
            P = _site_pauli(paulis[off:off+q]); off += q
            t = np.tensordot(E, A, axes=([1],[0]))
            t = np.tensordot(P, t, axes=([1],[1]))
            E = np.tensordot(A.conj(), t, axes=([0,1],[1,0]))
        return float(np.real(E[0,0]))


def sre_mcmc_ergodic(ev, n, n_samples, burn, seed=0, p_jump=0.15, jump_qubits=4):

    rng = np.random.default_rng(seed)
    P = np.zeros(n, dtype=np.int64); a = ev.expval(list(P))**2; ssum = 0.0
    for it in range(burn + n_samples):
        Pn = P.copy()
        if rng.random() < p_jump:
            qs = rng.choice(n, size=min(jump_qubits,n), replace=False)
            Pn[qs] = rng.integers(0,4,size=len(qs))
        else:
            Pn[rng.integers(n)] = rng.integers(4)
        b = ev.expval(list(Pn))**2
        if b >= a or rng.random() < b/a: P, a = Pn, b
        if it >= burn: ssum += a
    return -np.log2(ssum/n_samples)



#   M[p, (s,t)] = (P_p)_{s,t}          (s,t) flattened as 2*s + t
_M1 = np.array([[1, 0, 0,  1],          # I
                [0, 1, 1,  0],          # X
                [0,-1j,1j, 0],          # Y
                [1, 0, 0, -1]], dtype=complex)


def pauli_transform(G, q):

    rest = G.shape[2:]
    d = 1 << q
    # (s_1..s_q, t_1..t_q, rest) -> interleave to (s_1,t_1, s_2,t_2, ..., rest)
    t = G.reshape([2]*q + [2]*q + list(rest))
    order = []
    for i in range(q):
        order += [i, q+i]
    t = np.transpose(t, order + list(range(2*q, 2*q+len(rest))))
    t = t.reshape([4]*q + list(rest))
    for ax in range(q):                       # apply M on each qubit axis
        t = np.tensordot(_M1, t, axes=([1], [ax]))
        t = np.moveaxis(t, 0, ax)
    return t.reshape([4**q] + list(rest))


def pauli_transform_batch(G, q):
    """Vectorized Pauli transform for a batch of dressed-site transfer tensors.

    Parameters
    ----------
    G : ndarray, shape (B, 2**q, 2**q, chiR, chiR)
        One local transfer tensor per independent perfect-sampling walker.
    q : int
        Number of qubits in this dressed site.

    Returns
    -------
    W : ndarray, shape (B, 4**q, chiR, chiR)
        Pauli-transformed transfer matrices for all walkers and all local
        Pauli strings.  This is exactly the same transform as pauli_transform,
        with the batch axis carried through untouched.
    """
    batch = G.shape[0]
    rest = G.shape[3:]
    t = G.reshape([batch] + [2]*q + [2]*q + list(rest))
    order = [0]
    for i in range(q):
        order += [1+i, 1+q+i]
    order += list(range(1+2*q, 1+2*q+len(rest)))
    t = np.transpose(t, order)
    t = t.reshape([batch] + [4]*q + list(rest))
    for ax in range(q):
        pax = 1 + ax
        t = np.tensordot(_M1, t, axes=([1], [pax]))
        t = np.moveaxis(t, 0, pax)
    return t.reshape([batch, 4**q] + list(rest))


def _factorize_right_envs(R, verbose=False):
    """Numerically exact PSD factors for doubled right environments.

    For each R[k] interpreted as a matrix on the doubled bond, diagonalize the
    Hermitian part and form Y such that R ~= Y Y^dagger.  Only eigenvalues at
    the ordinary floating-point numerical-zero scale (dim*eps*lambda_max) are
    removed.  The factor is *used* only when it reduces the contracted
    dimension enough to be worthwhile; otherwise sampling keeps the dense R.

    Returns
    -------
    factors : list
        factors[k] is Y with shape (m, r), or None when dense R is preferable.
    info : list of dict
        numerical rank diagnostics for each environment.
    """
    factors = [None] * len(R)
    info = [None] * len(R)
    eps = np.finfo(float).eps
    for k in range(1, len(R)):  # R[0] is used only for the global norm check
        m = int(R[k].shape[0] * R[k].shape[1])
        Rm = R[k].reshape(m, m)
        if m == 1:
            val = max(float(np.real(Rm[0, 0])), 0.0)
            factors[k] = np.array([[np.sqrt(val)]], dtype=complex)
            info[k] = dict(dim=1, rank=1, tol=0.0, rel_discard=0.0,
                           report_ranks={rt: 1 for rt in ENV_RANK_REPORT_RTOLS},
                           use=True)
            continue

        herm_rel = float(np.linalg.norm(Rm - Rm.conj().T) / max(np.linalg.norm(Rm), 1e-300))
        H = 0.5 * (Rm + Rm.conj().T)
        w, U = np.linalg.eigh(H)
        # PSD up to roundoff.  Do not use a user-scale truncation here: only
        # remove eigenvalues indistinguishable from zero in floating arithmetic.
        wmax = max(float(w[-1]), 0.0)
        tol = ENV_RANK_EPS_MULT * m * eps * max(wmax, 1.0)
        keep = w > tol
        rank = int(np.count_nonzero(keep))
        report = {}
        if wmax > 0.0:
            for rt in ENV_RANK_REPORT_RTOLS:
                report[rt] = int(np.count_nonzero(w > rt*wmax))
        else:
            report = {rt: 0 for rt in ENV_RANK_REPORT_RTOLS}

        positive_sum = float(np.sum(np.clip(w, 0.0, None)))
        discarded = float(np.sum(np.clip(w[~keep], 0.0, None)))
        rel_discard = discarded/positive_sum if positive_sum > 0 else 0.0
        use = (herm_rel < 1e-10 and rank > 0 and rank <= ENV_FACTOR_MIN_SAVING*m)
        if use:
            factors[k] = U[:, keep] * np.sqrt(w[keep])[None, :]
        info[k] = dict(dim=m, rank=rank, tol=tol, rel_discard=rel_discard,
                       herm_rel=herm_rel, report_ranks=report, use=use)

        # Release eigensystem temporaries early; these can be large at chi=32.
        del U, w, H

    if verbose:
        print("      doubled-environment numerical ranks:", flush=True)
        for k in range(1, len(R)):
            d = info[k]
            rr = ", ".join(f"r@{rt:.0e}={d['report_ranks'][rt]}"
                           for rt in ENV_RANK_REPORT_RTOLS)
            mode = "factor" if d['use'] else "dense"
            print(f"        R[{k:2d}]: dim={d['dim']:4d}  rank={d['rank']:4d}  "
                  f"{rr}  discard={d['rel_discard']:.1e}  "
                  f"herm={d.get('herm_rel',0.0):.1e} -> {mode}", flush=True)
    return factors, info


def _quadratic_weights_batch(W, Rnext, Ynext=None):
    """All exact conditional weights, vectorized over walkers and Paulis.

    If a numerical-rank factor Y is available with R = Y Y^dagger, evaluate
        p = ||V Y||^2
    instead of the dense
        p = diag(V R V^dagger).
    This changes no target distribution beyond floating-point roundoff.
    """
    Bn, P, c1, c2 = W.shape
    m = c1*c2
    V = W.reshape(Bn*P, m)
    if Ynext is not None:
        VY = V @ Ynext
        out = np.sum(np.abs(VY)**2, axis=1).real.reshape(Bn, P)
    else:
        Rm = Rnext.reshape(m, m)
        VR = V @ Rm
        out = np.sum(VR * V.conj(), axis=1).real.reshape(Bn, P)
    # Only roundoff-size negative values are possible for a PSD environment.
    out[out < 0.0] = 0.0
    return out


def _categorical_rows(weights, rng):
    """Draw one categorical outcome from every row without Python loops."""
    totals = weights.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise RuntimeError("zero/nonfinite conditional in batched perfect sampler")
    cdf = np.cumsum(weights, axis=1)
    u = rng.random(weights.shape[0]) * totals
    idx = np.sum(cdf < u[:, None], axis=1)
    return np.minimum(idx, weights.shape[1]-1).astype(np.int64)


def sample_batch_full(Bs, qps, R, factors, rng, batch_size,
                      rao_blackwell_last=True):
    """Exact dressed-site perfect sampling for many independent walkers.

    The first L-1 dressed sites are sampled from the exact conditional Pauli
    distribution.  At the final dressed site, Rao--Blackwellization can replace
    the unnecessary final categorical draw by its exact conditional mean:

        E[ |c_P|^2 | prefix ] = sum_i x_i^2 / sum_i x_i,

    where x_i = |c_(prefix,i)|^2.  This is an unbiased estimator of the same
    E_Xi[|<P>|^2], but has variance no larger than sampling the last Pauli.
    """
    nb = int(batch_size)
    Lv = np.ones((nb, 1, 1), dtype=complex)
    L = len(Bs)
    for k, (B, q) in enumerate(zip(Bs, qps)):
        G1 = np.einsum('jae,asb->jsbe', Lv, B.conj(), optimize=True)
        G  = np.einsum('jsbe,etf->jstbf', G1, B, optimize=True)
        W  = pauli_transform_batch(G, q)
        weights = _quadratic_weights_batch(W, R[k+1], factors[k+1])

        if rao_blackwell_last and k == L-1:
            totals = weights.sum(axis=1)
            if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
                raise RuntimeError("zero/nonfinite final conditional")
            # At the last site R[L]=1, so weights are exactly the final
            # |c_(prefix,i)|^2 values.  Average the final draw analytically.
            return np.sum(weights*weights, axis=1) / totals

        idx = _categorical_rows(weights, rng)
        Lv = W[np.arange(nb), idx]
    return np.abs(Lv[:, 0, 0])**2


def _sample_chunk_batched(Bs, qps, R, factors, n_chunk, seed_sequence,
                          batch_size, rao_blackwell_last):
    """Worker kernel: several vectorized batches with an independent RNG."""
    rng = np.random.default_rng(seed_sequence)
    vals = np.empty(int(n_chunk), dtype=float)
    used = 0
    while used < n_chunk:
        nb = min(int(batch_size), int(n_chunk)-used)
        vals[used:used+nb] = sample_batch_full(
            Bs, qps, R, factors, rng, nb,
            rao_blackwell_last=rao_blackwell_last)
        used += nb
    return vals

def _X_tensor(B):
    """X[a,b,c,d] = sum_s conj(B[a,s,b]) B[c,s,d]."""
    return np.einsum('asb,csd->abcd', B.conj(), B, optimize=True)


def build_right_envs(Bs, qps):
    """R[k] has axes (a, a', c, c') -- the doubled bond of the c-MPS for the
       ket copy (a,a') and the bra copy (c,c').

       R_k[a,a',c,c'] = 2^q sum_{b,b',d,d'} X[a,b,c,d] conj(X[a',b',c',d'])
                                            R_{k+1}[b,b',d,d']

       which is the Pauli-completeness collapse -- no 4^q sum ever appears."""
    L = len(Bs)
    R = [None]*(L+1)
    R[L] = np.ones((1, 1, 1, 1), dtype=complex)
    for k in range(L-1, -1, -1):
        X = _X_tensor(Bs[k])                          # X[a,b,c,d]
        Z = np.einsum('abcd,bpdq->acpq', X, R[k+1], optimize=True)
        R[k] = (2.0**qps[k]) * np.einsum('acpq,epgq->aecg', Z, X.conj(),
                                         optimize=True)
    return R


def check_norm(R, n):
    got = float(np.real(R[0].reshape(-1)[0]))
    return got, 2.0**n, abs(got - 2.0**n)/2.0**n


def _prefix_marginal_weight(G, Rnext, Pprefix, m, q):
    """Exact marginal weight for a Pauli prefix on the first m qubits of one
    dressed site.  All 4**(q-m) unresolved Pauli suffixes are summed analytically
    using

        sum_Q Q[u,v] Q*[u',v'] = 2**(q-m) delta[u,u'] delta[v,v'].

    This is the key 4**q -> 4*q exact speedup; no change of sampling target.
    """
    dpre = 1 << m
    drem = 1 << (q-m)
    bdim, fdim = G.shape[2], G.shape[3]
    T = G.reshape(dpre, drem, dpre, drem, bdim, fdim)
    # H[u,v,b,f] = sum_{a,c} Pprefix[a,c] G[a,u,c,v,b,f]
    H = np.einsum('ac,aucvbf->uvbf', Pprefix, T, optimize=True)
    w = (2.0**(q-m)) * np.einsum('uvbf,uvdg,bfdg->',
                                  H, H.conj(), Rnext, optimize=True)
    return max(float(np.real(w)), 0.0), H


def sample_one_full(Bs, qps, R, rng):
    """Original dressed-site 4**q branching sampler, retained for validation."""
    L = len(Bs)
    Lv = np.ones((1, 1), dtype=complex)
    labels = []
    for k in range(L):
        B, q = Bs[k], qps[k]
        G1 = np.einsum('ae,asb->sbe', Lv, B.conj(), optimize=True)
        G  = np.einsum('sbe,etf->stbf', G1, B, optimize=True)
        W  = pauli_transform(G, q)
        p = np.einsum('pbf,pdg,bfdg->p', W, W.conj(), R[k+1], optimize=True)
        p = np.real(p); p[p < 0] = 0.0
        tot = p.sum()
        if tot <= 0:
            raise RuntimeError("zero conditional -- environments are wrong")
        i = int(rng.choice(len(p), p=p/tot))
        labels.append(i)
        Lv = W[i]
    return labels, float(abs(Lv.reshape(-1)[0])**2)


def sample_one_sequential(Bs, qps, R, rng):
    """Exact qubit-by-qubit perfect sampler inside each dressed site.

    Instead of evaluating all 4**q local Pauli strings, draw q successive
    four-way conditionals.  The marginal over the unresolved suffix is exact.
    """
    Lv = np.ones((1, 1), dtype=complex)
    labels = []
    for k, (B, q) in enumerate(zip(Bs, qps)):
        G1 = np.einsum('ae,asb->sbe', Lv, B.conj(), optimize=True)
        G  = np.einsum('sbe,etf->stbf', G1, B, optimize=True)

        Ppre = np.ones((1, 1), dtype=complex)
        label = 0
        Hchosen = None
        for m in range(1, q+1):
            weights = np.empty(4, dtype=float)
            prefixes = []
            Hs = []
            for p1 in range(4):
                P = np.kron(Ppre, _P1Q[p1])
                w, H = _prefix_marginal_weight(G, R[k+1], P, m, q)
                weights[p1] = w
                prefixes.append(P)
                Hs.append(H)
            tot = float(weights.sum())
            if not np.isfinite(tot) or tot <= 0:
                raise RuntimeError("zero/nonfinite sequential conditional")
            p1 = int(rng.choice(4, p=weights/tot))
            Ppre = prefixes[p1]
            Hchosen = Hs[p1]
            label = 4*label + p1

        labels.append(label)
        # At m=q the unresolved physical suffix has dimension one, and H is
        # exactly the conditioned transfer matrix W[label].
        Lv = Hchosen[0, 0]
    return labels, float(abs(Lv.reshape(-1)[0])**2)


def sample_one(Bs, qps, R, rng):
    return (sample_one_sequential(Bs, qps, R, rng) if PERFECT_SEQUENTIAL
            else sample_one_full(Bs, qps, R, rng))

def m2_perfect(Bs, qps, n_samples=2000, seed=0, verbose=False):
    """M_2 = -log2 E_Xi[<P>^2] with vectorized parallel perfect sampling.

    Three exact/floating-point-exact accelerations are combined:
      1. vectorized full-4**q walkers inside each batch,
      2. optional numerical-rank factorization R = Y Y^dagger,
      3. independent batches executed concurrently,
      4. Rao--Blackwellization of the final dressed-site Pauli draw.
    """
    n = int(sum(qps))

    t_env0 = time.perf_counter()
    R = build_right_envs(Bs, qps)
    t_env = time.perf_counter() - t_env0
    got, want, rel = check_norm(R, n)
    if rel > 1e-8:
        raise RuntimeError(f"sum_P |c_P|^2 = {got:.6e}, expected 2^n = {want:.6e}"
                           f"  (rel {rel:.2e}) -- environments are wrong")

    factors = [None] * len(R)
    factor_info = None
    t_fac = 0.0
    if ENV_FACTORIZE:
        tf0 = time.perf_counter()
        factors, factor_info = _factorize_right_envs(R, verbose=verbose)
        t_fac = time.perf_counter() - tf0

    batch = max(1, int(PERFECT_BATCH))
    workers = max(1, int(PERFECT_WORKERS))
    chunk = max(batch, int(PARALLEL_CHUNK_SAMPLES))
    # Keep chunk sizes aligned to the inner vectorized batch where possible.
    chunk = max(batch, (chunk // batch) * batch)

    if verbose:
        print(f"      [norm check] sum_P |c_P|^2 = {got:.6e} = 2^{n} ok", flush=True)
        facmsg = f"; env-factorization {t_fac:.2f}s" if ENV_FACTORIZE else ""
        print(f"      sampler = parallel batched full 4^q, batch={batch}, "
              f"workers={workers}, RB-last={RAO_BLACKWELL_LAST}; "
              f"right environments {t_env:.2f}s{facmsg}", flush=True)
        if workers > 1 and _TPSRE_THREADS != "1":
            print(f"      WARNING: PERFECT_WORKERS={workers} with TPSRE_THREADS="
                  f"{_TPSRE_THREADS}; use TPSRE_THREADS=1 to avoid BLAS "
                  f"oversubscription.", flush=True)

    # Adaptive stopping is kept serial-by-round when enabled so the stopping
    # decision is well defined.  Production default remains fixed n_samples.
    vals_parts = []
    used = 0
    t_samp0 = time.perf_counter()
    next_report = max(1, n_samples//5)
    report_at = next_report

    # Build enough independent chunks to expose concurrency without creating
    # one tiny Future for every inner batch.
    sizes = []
    remaining = int(n_samples)
    while remaining > 0:
        nc = min(chunk, remaining)
        sizes.append(nc)
        remaining -= nc
    root_ss = np.random.SeedSequence(seed)
    seeds = root_ss.spawn(len(sizes))

    def _consume(arr):
        nonlocal used, report_at
        vals_parts.append(arr)
        used += len(arr)
        if verbose and used >= report_at:
            vv = np.concatenate(vals_parts)
            mm = float(vv.mean())
            print(f"      perfect {used}/{n_samples}   M2 = {-np.log2(mm):.5f}", flush=True)
            while report_at <= used:
                report_at += next_report

    if workers == 1:
        for nc, ss in zip(sizes, seeds):
            arr = _sample_chunk_batched(Bs, qps, R, factors, nc, ss, batch,
                                        RAO_BLACKWELL_LAST)
            _consume(arr)
            if ADAPTIVE_PERFECT and used >= MIN_PERFECT_SAMPLES:
                vv = np.concatenate(vals_parts)
                mm = float(vv.mean())
                se = float(vv.std(ddof=1)/np.sqrt(len(vv)))
                err = se/(mm*np.log(2)) if mm > 0 else np.inf
                if err <= M2_TARGET_ERR:
                    break
    else:
        # Threads are intentional: NumPy/tensordot/matmul release the GIL, and
        # shared read-only MPS/environment arrays do not need process copies.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_sample_chunk_batched, Bs, qps, R, factors,
                                 nc, ss, batch, RAO_BLACKWELL_LAST)
                       for nc, ss in zip(sizes, seeds)]
            for fut in as_completed(futures):
                _consume(fut.result())
                # Fixed-sample production mode is the recommended parallel path.
                # We intentionally do not cancel already-running work for the
                # optional adaptive mode; any completed chunks are retained.

    t_samp = time.perf_counter() - t_samp0
    vals = np.concatenate(vals_parts)[:n_samples]
    mm = float(vals.mean())
    se = float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    if verbose and PROFILE_TIMINGS:
        rate = len(vals)/t_samp if t_samp > 0 else np.inf
        print(f"      timing: environments={t_env:.2f}s  factorization={t_fac:.2f}s  "
              f"sampling={t_samp:.2f}s  samples={len(vals)}  "
              f"rate={rate:.2f} samples/s", flush=True)
    return -np.log2(mm), se/(mm*np.log(2)), vals


def total_M2(As, qps, verbose=True):
    n = int(sum(qps))
    exact = np.nan

    # Only split dressed sites into individual qubits when a method actually
    # needs that representation.  For 4x2/4x4 perfect sampling it is wasted work.
    Q = None
    if n <= EXACT_MAX_QUBITS:
        Q = split_sites_to_qubits(As, qps)
        exact = m2_exact_fwht(mps_to_vector(Q), n)
        if not (VALIDATE_SAMPLER and SAMPLER != "none"):
            return exact, 0.0, "exact-FWHT"

    if SAMPLER == "none":
        return exact, 0.0, ("exact-FWHT" if np.isfinite(exact) else "skipped")

    if SAMPLER == "perfect":
        m, e, vals = m2_perfect(As, qps, n_samples=PERFECT_SAMPLES, seed=0,
                                verbose=verbose)
        how = (f"perfect-parallel(B={PERFECT_BATCH},W={PERFECT_WORKERS},"
               f"RB={int(RAO_BLACKWELL_LAST)}) x{len(vals)}")
    else:
        if Q is None:
            Q = split_sites_to_qubits(As, qps)
        need_gb = (2.0**n)*16/1e9
        if need_gb <= MEM_LIMIT_GB:
            ev = SparsePauliEval(mps_to_vector(Q), n); how = "dense"
        else:
            ev = MPSPauliEval(Q, qps); how = "MPS-contraction"
            if verbose: log(f"      dense state would need {need_gb:.1f} GB -> {how}")
        vals = [sre_mcmc_ergodic(ev, n, MC_SAMPLES, MC_BURN, seed=sd)
                for sd in MC_SEEDS]
        m, e = float(np.mean(vals)), float(np.std(vals, ddof=1)/np.sqrt(len(vals)))

    if np.isfinite(exact):
        dev = abs(m-exact)/max(e, 1e-12)
        rel = abs(m-exact)/max(abs(exact), 1e-12)
        bad = (dev > 3) or (rel > 0.10)
        log(f"      [validate] exact {exact:.6f}  vs  sampled {m:.6f} +/- {e:.6f}"
            f"   ({dev:.1f} sigma, {100*rel:.1f}% off)")
        if bad:
            log("      *** SAMPLER DISAGREES WITH EXACT BENCHMARK.")
            log("      *** Increase samples and inspect convergence before large runs.")
        return exact, 0.0, f"exact-FWHT ({'BIASED' if bad else 'ok'} sampler, {dev:.1f}s)"
    return m, e, how

def schmidt_values(psi, bond):
    """Normalized Schmidt values across a finite-MPS bond."""
    sv = psi.get_SL(bond) if hasattr(psi, 'get_SL') else psi.get_SV(bond)
    sv = np.asarray(sv, dtype=float)
    nrm = np.linalg.norm(sv)
    if nrm == 0.0:
        raise ValueError(f"zero Schmidt norm at bond {bond}")
    return sv / nrm

def entanglement_entropy(sv):
    """Von Neumann entropy (bits) from normalized Schmidt values."""
    p = np.asarray(sv, dtype=float)**2
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log2(p)))

def antiflatness(sv):
    """F = Tr(rho^3) - (Tr rho^2)^2; zero for a flat Schmidt spectrum."""
    p = np.asarray(sv, dtype=float)**2
    p = p / p.sum()
    return float(np.sum(p**3) - np.sum(p**2)**2)

def antiflatness_charge_resolved(psi, bond):
    """Charge-resolved antiflatness used for the gauge-invariant lower bound."""
    sv = schmidt_values(psi, bond)
    B = psi.get_B(bond, form='B')
    leg = B.get_leg('vL')
    if leg.chinfo.qnumber == 0:
        return antiflatness(sv)
    F = 0.0
    for bi in range(leg.block_number):
        blk = sv[leg.slices[bi]:leg.slices[bi + 1]]**2
        pQ = blk.sum()
        if pQ < 1e-30:
            continue
        r = blk / pQ
        F += pQ**2 * (np.sum(r**3) - np.sum(r**2)**2)
    return float(F)

def bipartite_scan(psi):
    out = []
    for b in range(1, psi.L):
        sv = schmidt_values(psi, b)
        F  = antiflatness(sv)
        out.append({'bond': b, 'S': entanglement_entropy(sv), 'F': F, '4F': 4*F,
                    'F_Q': antiflatness_charge_resolved(psi, b),
                    'chi': int(len(sv))})
    return out


def donnelly_ee_middle_cut(psi, Lx, Ly, eps=1e-10, max_sectors=8192):
    """Gauge-invariant (Donnelly) entanglement entropy at the middle VERTICAL cut,
    resolved by the full tuple of the Ly crossing-link SU(2) irreps -- the 2D
    analogue of the 1D boundary_irrep_resolved_EE (Jha et al.; prescription of
    Donnelly, Phys. Rev. D 85, 085004).  All entropies are in BITS (log2),
    consistent with the rest of the code.

    Column-major site order (site = jx*Ly + jy) makes the MPS bond at jx0*Ly a
    clean vertical cut between columns jx0-1 and jx0.  Its boundary is the Ly
    horizontal links (jx0-1,jy)--(jx0,jy); each link's SU(2) irrep j is read off
    the (diagonal) leg Casimir T2_px of the left site (T^2 = j(j+1)).  A sector is
    the tuple (j_0,...,j_{Ly-1}); projecting onto it gives p_sector and the
    conditional entropy S_sector.  Then
        S = -sum p_s log2 p_s + sum p_s S_s + sum p_s (sum_links log2(2j+1))
    and S_total = S_classical + S_quantum + S_edge is the gauge-invariant EE
    (with S_standard = S_classical + S_quantum the plain dressed-space entropy).
    Also returns the SU(2) irrep-resolved antiflatness F_GI (a *gauge-invariant*
    diagnostic; the reported lower bound on the whole-state M_2^NL remains the
    whole-state 4F)."""
    import itertools
    import tenpy.linalg.np_conserved as npc
    if Lx < 2:
        return None
    jx0 = Lx // 2
    cut = jx0 * Ly
    S_standard = entanglement_entropy(schmidt_values(psi, cut))    # bits (log2)

    def _proj(site, idxs):
        d = site.dim; mat = np.zeros((d, d))
        for a in idxs:
            mat[a, a] = 1.0
        return npc.Array.from_ndarray(mat, [site.leg, site.leg.conj()],
                                      labels=['p', 'p*'])

    rows = []                                   # per crossing link: (site, {j: idxs})
    for jy in range(Ly):
        sL = (jx0 - 1) * Ly + jy
        site = psi.sites[sL]
        diag = np.diag(site.get_op('T2_px').to_ndarray().real)
        jj = np.round(((-1 + np.sqrt(np.maximum(0.0, 1 + 4*diag))) / 2) * 2) / 2
        g = {}
        for a in range(site.dim):
            g.setdefault(float(jj[a]), []).append(a)
        rows.append((sL, site, g))

    # per-link marginals -> only keep irreps with non-negligible weight
    allowed = []
    for (sL, site, g) in rows:
        keep = [j for j, idxs in g.items()
                if float(np.real(psi.expectation_value(_proj(site, idxs),
                                                       sites=[sL])[0])) > eps]
        allowed.append(sorted(keep))
    ncomb = int(np.prod([max(1, len(a)) for a in allowed]))
    if ncomb > max_sectors:
        return {'cut_bond': cut, 'S_standard': S_standard, 'n_sectors': ncomb,
                'error': f'{ncomb} sectors exceed max_sectors={max_sectors}'}

    S_classical = S_quantum = S_edge = p_norm = 0.0
    F_GI = 0.0                          # SU(2) irrep-resolved antiflatness
    blocks = {}
    for combo in itertools.product(*allowed):
        pc = psi.copy()
        for (sL, site, g), j in zip(rows, combo):
            pc.apply_local_op(sL, _proj(site, g[j]), unitary=False,
                              renormalize=False)
        p_s = float(np.real(pc.overlap(pc)))
        if p_s <= eps:
            continue
        pc.canonical_form_finite(renormalize=True)
        sv_s = schmidt_values(pc, cut)
        S_s = entanglement_entropy(sv_s)               # bits (log2)
        # antiflatness within this superselection sector (normalized rho_A^(s)):
        rho = sv_s**2
        F_s = float(np.sum(rho**3) - np.sum(rho**2)**2)
        edge = float(sum(np.log2(2*j + 1) for j in combo))     # bits
        blocks[tuple(combo)] = (p_s, S_s, F_s, edge)
        p_norm += p_s
        S_classical -= p_s * np.log2(p_s)              # bits
        S_quantum += p_s * S_s
        S_edge += p_s * edge
        F_GI += p_s**2 * F_s          # F_GI = sum_s p_s^2 F_A(rho_A^(s))
    return {'cut_bond': cut, 'S_standard': S_standard,
            'S_classical': S_classical, 'S_quantum': S_quantum, 'S_edge': S_edge,
            'S_total': S_classical + S_quantum + S_edge,
            'F_GI': F_GI, '4F_GI': 4.0 * F_GI,
            'p_norm': p_norm, 'n_sectors': len(blocks), 'blocks': blocks}

def nonlocal_magic_reference(schmidt_vals, tol=1e-12, max_rank=None):
    """Conjectured nonlocal 2-SRE of a bipartite cut from its entanglement
    spectrum, via the Schmidt reference state (Liu & Cui, arXiv:2607.10714).

    Given the (normalised) Schmidt values s_i of a cut, the sorted Schmidt
    probabilities lambda_i = s_i^2 are placed on a canonical reference state
        |psi~> = sum_i sqrt(lambda_i) |i>_A~ |i>_B~
    on 2*ceil(log2 chi) qubits (a computational-basis encoding of the ordered
    spectrum).  Its ordinary 2-SRE equals the paper's Eq. (10) and, by the
    conjecture, the nonlocal SRE M_2^NL of the original cut -- the magic that no
    local unitary on either side can remove.  It obeys  4F <= M_2^NL <= M_2.

    Returns dict with 'M2_NL', 'eta' (discarded Schmidt weight from truncation),
    'chi_kept', and 'n_ref' (qubits in the reference state).  For large ranks set
    max_rank to keep only the largest Schmidt probabilities; the paper bounds the
    truncation error by the discarded weight eta (Eq. 11-12), so monitor eta."""
    lam = np.asarray(schmidt_vals, dtype=float)**2
    lam = np.sort(lam[lam > tol])[::-1]
    if lam.size == 0:
        return {'M2_NL': 0.0, 'eta': 0.0, 'chi_kept': 0, 'n_ref': 0}
    lam = lam / lam.sum()
    eta = 0.0
    if max_rank is not None and lam.size > int(max_rank):
        eta = float(lam[int(max_rank):].sum())
        lam = lam[:int(max_rank)]
        lam = lam / lam.sum()
    chi = lam.size
    q = max(1, int(np.ceil(np.log2(chi))))          # qubits per side
    d = 1 << q
    lam_pad = np.zeros(d, dtype=float); lam_pad[:chi] = lam
    amp = np.sqrt(lam_pad)
    ref = np.zeros(d * d, dtype=complex)            # |psi~> = sum_i amp_i |i,i>
    idx = np.arange(d)
    ref[idx * d + idx] = amp
    m2nl = float(m2_exact_fwht(ref, 2 * q))
    return {'M2_NL': max(0.0, m2nl), 'eta': eta, 'chi_kept': int(chi),
            'n_ref': int(2 * q)}


def nonlocal_magic_minimize(schmidt_vals, keep=8, restarts=3, seed=0,
                            maxiter=250, tol=1e-12):
    """INDEPENDENT nonlocal 2-SRE by the *definition*: minimize the ordinary SRE
    over local unitaries,  M_2^NL = min_{U_A,U_B} M_2(U_A x U_B |psi~>)  (Cao et
    al. arXiv:2502.06393; White et al. arXiv:2607.07808).  This does NOT use the
    reference-state conjecture, so comparing it to nonlocal_magic_reference() tests
    that conjecture directly.

    The spectrum is truncated to the largest `keep` Schmidt probabilities; the
    reference state (restart 0 = identity) is always in the search space, so the
    returned minimum is <= the reference value.  Agreement (min == reference)
    supports the conjecture; finding min < reference would refute it.  Returns a
    dict, or None if SciPy is unavailable."""
    try:
        from scipy.linalg import expm
        from scipy.optimize import minimize
    except Exception:
        return None
    lam = np.asarray(schmidt_vals, dtype=float)**2
    lam = np.sort(lam[lam > tol])[::-1]
    if lam.size == 0:
        return None
    lam = lam / lam.sum(); eta = 0.0
    if lam.size > keep:
        eta = float(lam[keep:].sum()); lam = lam[:keep]; lam = lam / lam.sum()
    chi = lam.size; m = max(1, int(np.ceil(np.log2(chi)))); d = 1 << m
    Lp = np.zeros(d); Lp[:chi] = lam
    P0 = np.diag(np.sqrt(Lp)).astype(complex)          # reference state (U = I)
    iu = np.triu_indices(d, 1); nu = len(iu[0]); npar = d + 2 * nu
    def herm(p):
        H = np.zeros((d, d), dtype=complex)
        H[np.diag_indices(d)] = p[:d]
        H[iu] = p[d:d+nu] + 1j * p[d+nu:d+2*nu]
        return H + H.conj().T - np.diag(np.diag(H).real)
    ref = float(m2_exact_fwht(P0.reshape(-1), 2 * m))
    def f(x):
        UA = expm(1j * herm(x[:npar])); UB = expm(1j * herm(x[npar:]))
        v = (UA @ P0 @ UB.T).reshape(-1); v = v / np.linalg.norm(v)
        return float(m2_exact_fwht(v, 2 * m))
    rng = np.random.default_rng(seed); best = ref
    for r in range(max(1, restarts)):
        x0 = np.zeros(2 * npar) if r == 0 else 0.6 * rng.standard_normal(2 * npar)
        res = minimize(f, x0, method='L-BFGS-B', options={'maxiter': maxiter})
        if res.fun < best:
            best = float(res.fun)
    return {'M2_NL_min': max(0.0, best), 'M2_NL_ref': ref, 'eta': eta,
            'chi_kept': int(chi), 'n_ref': int(2 * m)}


_SRE_AUTOGRAD_CONSTS = {}
def _sre_autograd_consts(N):
    """Cached constants for the differentiable full-x-sweep 2-SRE on N qubits:
    the (unnormalised) Hadamard matrix, the b XOR x index map, and the
    i^{|x wedge z|} phase table."""
    c = _SRE_AUTOGRAD_CONSTS.get(N)
    if c is None:
        d = 1 << N
        H = np.array([[1.0]])
        for _ in range(N):
            H = np.block([[H, H], [H, -H]])
        idx = np.arange(d)
        xor = (idx[:, None] ^ idx[None, :]).reshape(-1)          # [x,b] -> b^x
        pop = np.array([bin(i).count('1') for i in range(d)])
        phase = (1j) ** pop[(idx[:, None] & idx[None, :])]       # [x,z]
        c = (d, H.astype(complex), xor, phase)
        _SRE_AUTOGRAD_CONSTS[N] = c
    return c


def nonlocal_magic_minimize_riemannian(schmidt_vals, keep=8, seed=0,
                                       max_iterations=250, tol=1e-12):
    """Same nonlocal 2-SRE cross-check as nonlocal_magic_minimize(), but using the
    Riemannian optimizer of arXiv:2607.10714: minimize M_2 over local unitaries
    with manifold optimization (pymanopt ConjugateGradient) on the product of two
    complex unitary manifolds U(2^m) x U(2^m), autodiff gradients (autograd).
    Optimizing over U(2^m) is equivalent to the paper's complex-Stiefel reduction
    (the extra columns carry vanishing Schmidt weight).  Returns the same dict as
    nonlocal_magic_minimize(), or None if pymanopt/autograd are unavailable."""
    try:
        import autograd.numpy as anp
        import pymanopt
        from pymanopt.manifolds import UnitaryGroup, Product
        from pymanopt.optimizers import ConjugateGradient
    except Exception:
        return None
    lam = np.asarray(schmidt_vals, dtype=float) ** 2
    lam = np.sort(lam[lam > tol])[::-1]
    if lam.size == 0:
        return None
    lam = lam / lam.sum(); eta = 0.0
    if lam.size > keep:
        eta = float(lam[keep:].sum()); lam = lam[:keep]; lam = lam / lam.sum()
    chi = lam.size; m = max(1, int(np.ceil(np.log2(chi)))); d = 1 << m; N = 2 * m
    Lp = np.zeros(d); Lp[:chi] = lam
    lamv = np.sqrt(Lp)
    _, Hmat, xor, phase = _sre_autograd_consts(N)
    Hc = anp.array(Hmat); ph = anp.array(phase); lam_a = anp.array(lamv)

    def sre_of_state(v):                       # differentiable exact 2-SRE
        v = v / anp.sqrt(anp.sum(anp.abs(v) ** 2))
        vbx = v[xor].reshape(1 << N, 1 << N)   # [x,b] = v[b^x]
        M = v[None, :] * anp.conj(vbx)         # [x,b] = v[b] v*[b^x]
        C = M @ Hc                             # FWHT over b -> [x,z]
        re = anp.real(ph * C)
        return -anp.log2(anp.sum(re ** 4) / (1 << N))

    man = Product([UnitaryGroup(d), UnitaryGroup(d)])

    @pymanopt.function.autograd(man)
    def cost(UA, UB):
        Psi = UA @ (lam_a[:, None] * UB.T)     # [a,b] = sum_i UA[a,i] lam_i UB[b,i]
        return sre_of_state(Psi.reshape(-1))

    ref = float(m2_exact_fwht((np.diag(lamv)).reshape(-1), N))
    I = np.eye(d, dtype=complex)
    try:
        opt = ConjugateGradient(max_iterations=int(max_iterations), verbosity=0)
        res = opt.run(pymanopt.Problem(man, cost), initial_point=[I, I])
        best = float(res.cost)
    except Exception:
        return None
    return {'M2_NL_min': max(0.0, min(best, ref)), 'M2_NL_ref': ref, 'eta': eta,
            'chi_kept': int(chi), 'n_ref': int(N)}

# ---- Creutz ratio and roughening susceptibilities ---------------------------
def creutz_11(ReBp):
    """Smallest Creutz ratio  chi(1,1) = -ln W(1,1) = -ln Re<B_p>  (string tension
    at scale 1; W(1,1) is the plaquette).  NOTE: chi(1,1) is MONOTONIC in g^2 --
    it does not peak.  In the hardcore-gluon (j_max=1/2) truncation the plaquette
    is the only Wilson loop faithfully represented; larger loops W(2,2), ... need
    higher electric representations, so their Creutz ratios are not reliable here.
    The roughening/magneto-electric crossover instead shows up as a PEAK in the
    susceptibilities below."""
    return float(-np.log(max(ReBp, 1e-15))) if np.isfinite(ReBp) else float('nan')

def roughening_diagnostics(rows):
    """Finite-difference susceptibilities that peak at the magneto-electric /
    roughening crossover: the 'specific heat'  C = d^2 eps / d(ln g^2)^2  and the
    plaquette susceptibility  dP_bar / d(ln g^2)  (nonuniform central differences
    on the interior g^2 grid).  A denser g^2 grid near the transition sharpens
    the peak."""
    if len(rows) < 3:
        return []
    x = np.log(np.array([r['g2'] for r in rows]))
    eps = np.array([r['eps'] for r in rows])
    P = np.array([r['ReBp'] for r in rows])
    out = []
    for i in range(1, len(rows)-1):
        h1, h2 = x[i]-x[i-1], x[i+1]-x[i]
        C = 2.0*(eps[i-1]/(h1*(h1+h2)) - eps[i]/(h1*h2) + eps[i+1]/(h2*(h1+h2)))
        dP = (P[i+1]-P[i-1])/(x[i+1]-x[i-1])
        out.append({'g2': rows[i]['g2'], 'C': float(C), 'dP_dlng2': float(dP),
                    'chi11': rows[i]['chi11']})
    return out



# ---- self-test of the Hamiltonian against independent reference values -------
# Reference energies for the 2x2 OBC lattice, b=0, m=1 (total energies, NOT per site), obtained from two
# independent constructions that do not use the rishon regrouping: (i) the original link basis with the
# 5-state truncated U of Cataldi et al. (B33) and Jordan-Wigner matter fermions, (ii) a global-JW rishon
# construction with Tr(U U U^+ U^+) evaluated literally.  Both agree to all digits shown.
_SELFTEST_REF = {0.2: (-9.128762, -5.441250, -5.313481, -5.313481),
                 1.0: (-5.126700, -2.555705, -2.335795, -2.334659)}

def selftest_hamiltonian(verbose=True):
    """Build the 2x2 Hamiltonian densely from the same on-site operators used by build_model(), project it on
    the physical space (link law + b=0, dimension 38) and compare the lowest levels with _SELFTEST_REF.
    Pure NumPy; runs in a few seconds.  Returns True on success."""
    Lx, Ly = 2, 2
    coords = [(jx, jy) for jx in range(Lx) for jy in range(Ly)]
    Ms = {c: site_basis(frozen_legs_of(Lx, Ly, *c)) for c in coords}
    dims = [Ms[c].shape[1] for c in coords]
    red = lambda c, O: Ms[c].conj().T @ O @ Ms[c]
    def emb(ops):
        return reduce(np.kron, [ops.get(c, np.eye(dims[i])) for i, c in enumerate(coords)])
    Dtot = int(np.prod(dims))
    def H_of(g2, mass=1.0):
        B = -1.0/(2.0*g2); H = np.zeros((Dtot, Dtot), complex)
        for (x, y) in coords:
            u = (x, y)
            H += mass*((-1)**(x+y))*emb({u: red(u, N_raw)}) + (g2/2)*emb({u: red(u, gamma)})
            if x+1 < Lx:
                t = -1j*T_HOP*emb({u: red(u, Qd_raw['px']), (x+1, y): red((x+1, y), Q_raw['mx'])}); H += t + t.conj().T
            if y+1 < Ly:
                t = -T_HOP*((-1)**(x+y))*emb({u: red(u, Qd_raw['py']), (x, y+1): red((x, y+1), Q_raw['my'])}); H += t + t.conj().T
            if x+1 < Lx and y+1 < Ly:
                t = B*emb({u: red(u, C_pp), (x+1, y): red((x+1, y), C_mp),
                           (x, y+1): red((x, y+1), C_pm), (x+1, y+1): red((x+1, y+1), C_mm)}); H += t + t.conj().T
        return H
    links = [((0,0),'px',(1,0),'mx'), ((0,1),'px',(1,1),'mx'), ((0,0),'py',(0,1),'my'), ((1,0),'py',(1,1),'my')]
    Lc = sum((emb({a: red(a, T2_leg[la])}) - emb({b: red(b, T2_leg[lb])})) @
             (emb({a: red(a, T2_leg[la])}) - emb({b: red(b, T2_leg[lb])})) for a, la, b, lb in links)
    Ntot = sum(emb({c: red(c, N_raw)}) for c in coords)
    w, v = np.linalg.eigh(Lc + (Ntot - 4*np.eye(Dtot)) @ (Ntot - 4*np.eye(Dtot)))
    W = v[:, np.abs(w) < 1e-9]
    ok = W.shape[1] == 38
    if verbose:
        log(f"  [selftest] physical dimension (link law + b=0) = {W.shape[1]} (expected 38)")
    for g2, ref in _SELFTEST_REF.items():
        e = np.linalg.eigvalsh(W.conj().T @ H_of(g2) @ W)[:4]
        good = np.allclose(e, ref, atol=2e-6)
        ok &= good
        if verbose:
            log(f"  [selftest] g^2={g2}: E0..E3 = {np.round(e, 6)}  reference {ref}  -> {'ok' if good else 'MISMATCH'}")
    if verbose:
        log("  [selftest] " + ("PASSED" if ok else "FAILED -- Hamiltonian does not match Eqs. (5)-(6) of the paper"))
    return ok

# ---- ground-state checkpointing ---------------------------------------------
def _mtag(mass=None):
    """Mass label used in file names: 1.0 -> '1', 0.2 -> '0.2'."""
    return f"{float(MASS if mass is None else mass):g}"

def _ckpt_path(lattice, g2, chi):
    """gs_<lattice>_m<mass>_g<g2>_chi<chi>.pkl -- the mass is part of the name so
    the m=1 and m=0.2 scans never read or overwrite each other's states (and old
    files without '_m' are never picked up)."""
    return os.path.join(CKPT_DIR, f"gs_{lattice}_m{_mtag()}_g{g2}_chi{chi}.pkl")

def save_gs(path, r):
    """Persist ONLY the ground state + cheap metadata.  Observables and magic
    are (re)computed on load, so a change in their definitions always
    propagates instead of being frozen into old checkpoints.  Written to a
    temporary file first, so a job killed mid-write cannot leave a truncated
    checkpoint behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keep = {k: r[k] for k in ('psi', 'qps', 'perm', 'eps', 'chi', 'S_tot')}
    if 'walltime_gs' in r:
        keep['walltime_gs'] = r['walltime_gs']
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(keep, f)
    os.replace(tmp, path)

SITE_CHECK_TOL = 5e-3     # |<H>/N - stored eps| per site: truncation noise <~1e-4, wrong labels >~3e-2

def _eps_with_sites(psi, sites, Lx, Ly, g2):
    """<H_phys>/N of psi, with the Hamiltonian built from the given Site objects."""
    lat = TeNPyLattice([1, 1], sites, bc='open', bc_MPS='finite')
    model = build_model(lat, Lx, Ly, g2, MASS, True)
    Hpen = build_penalty_mpo(lat, Lx, Ly)
    E = float(np.real(model.H_MPO.expectation_value(psi)))
    return (E - float(np.real(Hpen.expectation_value(psi)))) / (Lx*Ly)

def load_gs(path, Lx, Ly, g2=None):
    """Load a ground-state checkpoint and recompute the gauge observables.
       With g2 given, first make sure psi.sites match the basis its tensors were
       optimized in (checkpoints written before the site-relabelling fix in
       run_plain_gs may carry the sites of the seed's machine): <H> built from the
       pickled sites, else from this process's sites, must reproduce the stored DMRG
       energy.  If neither does, return None so the point is recomputed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            r = pickle.load(f)
    except Exception as exc:              # unreadable -> recompute this point
        log(f"  [checkpoint] could not read {path} ({exc!r}); recomputing")
        return None
    if g2 is not None and np.isfinite(r.get('eps', np.nan)):
        psi = r['psi']
        e_pick = _eps_with_sites(psi, psi.sites, Lx, Ly, g2)
        if abs(e_pick - r['eps']) > SITE_CHECK_TOL:
            here = list(build_sites(Lx, Ly)[0])
            e_here = _eps_with_sites(psi, here, Lx, Ly, g2)
            if abs(e_here - r['eps']) <= SITE_CHECK_TOL:
                log(f"  [checkpoint] {path}: pickled sites do not match the tensors "
                    f"(<H>/N={e_pick:.8f} vs eps={r['eps']:.8f}); this process's sites "
                    f"do (<H>/N={e_here:.8f}) -> relabelled")
                psi.sites = here
            else:
                log(f"  [checkpoint] {path}: neither pickled nor local sites reproduce "
                    f"eps={r['eps']:.8f} (<H>/N = {e_pick:.8f} / {e_here:.8f}): written in "
                    f"another site basis -> recomputing this point")
                return None
    lat = TeNPyLattice([1, 1], r['psi'].sites, bc='open', bc_MPS='finite')
    r.update(measure_gauge_observables(r['psi'], lat, Lx, Ly))
    return r

# ---- per-sweep DMRG checkpoints (a g^2 point may span several jobs) ----------
class _OutOfTime(Exception):
    """Raised from the sweep hook when the next sweep would not fit before DEADLINE."""

class _SeedNotReady(Exception):
    """--seed-chi state for this g^2 does not exist yet."""

def _partial_path(g2):
    return _ckpt_path(LATTICE, g2, CHI)[:-len(".pkl")] + ".partial.pkl"

def _atomic_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def _load_partial(g2):
    path = _partial_path(g2)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        log(f"  [checkpoint] could not read {path} ({exc!r}); ignoring it")
        return None

def _make_sweep_hook(g2, t_prior, t_start):
    """Called by TeNPy after every completed sweep (except the last): saves the
    resume data of the running DMRG and stops (raises _OutOfTime) if the next
    sweep would not finish before DEADLINE."""
    path = _partial_path(g2)
    st = {'last': t_start, 'dt_max': 0.0}
    def hook(eng):
        now = time.time()
        dt = now - st['last']; st['last'] = now; st['dt_max'] = max(st['dt_max'], dt)
        mx = eng.mixer
        mixer = None
        if mx is not None:
            left = mx.sweep_activated + mx.disable_after - eng.sweeps \
                   if mx.disable_after is not None else None
            if left is None or left > 0:
                mixer = {'amplitude': mx.amplitude, 'decay': mx.decay, 'disable_after': left}
        if SAVE_GS:
            _atomic_pickle({'data': eng.get_resume_data(), 'mixer': mixer,
                            'min_sweeps': int(eng.options.get('min_sweeps', 1, int)),
                            'sweeps': eng.sweeps, 'walltime_gs': t_prior + (now - t_start),
                            'g2': g2, 'chi': CHI, 'mass': MASS, 'lattice': LATTICE}, path)
        ss = eng.sweep_stats
        mix_s = "off" if mixer is None else "%.1e" % mixer['amplitude']
        log(f"      sweep {eng.sweeps:>3}: E={ss['E'][-1]:.12f}  dE={ss['Delta_E'][-1]:+.2e}"
            f"  chi={max(eng.psi.chi)}  mixer={mix_s}  ({dt/60:.1f} min)  [saved]")
        if DEADLINE is not None and now + 1.2*st['dt_max'] > DEADLINE:
            raise _OutOfTime(f"stopped after sweep {eng.sweeps}: next sweep (~{st['dt_max']/60:.0f} min)"
                             f" would pass the deadline; resume data in {path}")
    return hook

# ---- output JSON: one file per (lattice, chi, mass), updated after every g^2 --
def _json_path(lattice=None, chi=None, mass=None):
    """nonlocal_magic_<lattice>_D<chi>_m<mass>.json  (sample: nonlocal_magic_6x6_D128.json)."""
    return (f"nonlocal_magic_{LATTICE if lattice is None else lattice}_"
            f"D{CHI if chi is None else chi}_m{_mtag(mass)}.json")

def _read_rows(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log(f"  [json] could not read {path} ({exc!r}); treating as empty")
        return []

def _done_g2(path):
    """g^2 values that already have a row in this JSON file."""
    return {float(x['g2']) for x in _read_rows(path) if 'g2' in x}

def _point_seconds_estimate(timed_this_run):
    """Expected wall time of the next g^2 point: the largest per-point walltime
    recorded for this lattice and chi (any mass) or measured in this run; else
    POINT_HOURS_GUESS; else 0 (unknown -> always start)."""
    vals = list(timed_this_run)
    for fn in glob.glob(f"nonlocal_magic_{LATTICE}_D{CHI}_m*.json"):
        vals += [float(x['walltime']) for x in _read_rows(fn)
                 if isinstance(x.get('walltime'), (int, float)) and np.isfinite(x['walltime'])]
    if vals:
        return max(vals)
    return 3600.0*POINT_HOURS_GUESS if POINT_HOURS_GUESS else 0.0

# ---- warm-start seed selection ----------------------------------------------
def _checkpoints_on_disk(lattice, chi):
    """{g^2: path} for every ground-state checkpoint in CKPT_DIR with this lattice,
    MASS and bond dimension (file names as written by _ckpt_path)."""
    out = {}
    if not os.path.isdir(CKPT_DIR):
        return out
    pat = re.compile(rf"^gs_{re.escape(str(lattice))}_m{re.escape(_mtag())}"
                     rf"_g(.+)_chi{int(chi)}\.pkl$")
    for fn in os.listdir(CKPT_DIR):
        m = pat.match(fn)
        if m is None:
            continue
        try:
            out[float(m.group(1))] = os.path.join(CKPT_DIR, fn)
        except ValueError:
            pass
    return out

def _load_seed_psi(path):
    """Read only the MPS (+ eps for the log) from a checkpoint.  A seed does not
    need the gauge observables, so load_gs's re-measurement is skipped."""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['psi'], d.get('eps', float('nan'))

def _warm_start_seed(g2, prev):
    """Choose the DMRG initial state for coupling g2.
       prev = (g2_prev, psi_prev) of the point handled just before in THIS run
              (None for the first point).
       Returns (psi0, message); psi0 = None means a fresh product-state start."""
    if not WARM_START_G2:
        return None, "WARM_START_G2=False -> fresh start"
    if SEED_CHI is not None:
        # segment of a bond-dimension ladder: continuation INSIDE this --g2 list,
        # the first point of the list from the chi=SEED_CHI state at the same g^2
        lo = min(G2_VALUES)
        if prev is not None and lo <= prev[0] < g2:
            return prev[1], f"seed = g^2={prev[0]} (previous point of this segment, in memory)"
        inseg = {g: p for g, p in _checkpoints_on_disk(LATTICE, CHI).items() if lo <= g < g2}
        for gs in sorted(inseg, reverse=True):
            try:
                psi, e = _load_seed_psi(inseg[gs])
                return psi, f"seed = {inseg[gs]}  (previous point of this segment, eps={e:.10f})"
            except Exception as exc:
                log(f"    [warm-start] could not read {inseg[gs]} ({exc!r}); trying next")
        path = _ckpt_path(LATTICE, g2, SEED_CHI)
        if os.path.exists(path):
            psi, e = _load_seed_psi(path)
            return psi, (f"seed = chi={SEED_CHI} ground state at the same g^2: {path}  "
                         f"(eps={e:.10f})")
        raise _SeedNotReady(path)
    if COLD_START_G2 is not None and g2 <= COLD_START_G2:
        return None, (f"g^2 <= COLD_START_G2={COLD_START_G2} -> fresh start "
                      f"(no seed search)")
    if WARM_START_SEED_PICK not in ("nearest", "smallest"):
        raise ValueError(f"unknown WARM_START_SEED_PICK={WARM_START_SEED_PICK!r} "
                         "(expected 'nearest' or 'smallest')")

    # manual override (old behaviour): only for the first point of the run
    if prev is None and WARM_START_SEED_G2 is not None:
        path = _ckpt_path(LATTICE, WARM_START_SEED_G2, CHI)
        if os.path.exists(path):
            psi, e = _load_seed_psi(path)
            return psi, (f"seed = {path}  (manual WARM_START_SEED_G2="
                         f"{WARM_START_SEED_G2}, eps={e:.10f})")
        log(f"    [warm-start] manual seed {path} not found; searching automatically")

    # candidates: checkpoints on disk with smaller g^2, plus the point computed
    # just before in this run (None -> take it from memory, no file needed)
    cands = {g: p for g, p in _checkpoints_on_disk(LATTICE, CHI).items() if g < g2}
    if prev is not None and prev[0] < g2:
        cands[prev[0]] = None
    if not cands:
        return None, (f"no checkpoint with g^2 < {g2} ({LATTICE}, m={MASS}, chi={CHI}, "
                      f"dir={CKPT_DIR}) -> fresh start")
    for gs in sorted(cands, reverse=(WARM_START_SEED_PICK == "nearest")):
        if cands[gs] is None:
            return prev[1], f"seed = g^2={gs} computed just before in this run (in memory)"
        try:
            psi, e = _load_seed_psi(cands[gs])
            return psi, f"seed = {cands[gs]}  (g^2={gs}, eps={e:.10f})"
        except Exception as exc:          # e.g. truncated pickle from a killed job
            log(f"    [warm-start] could not read {cands[gs]} ({exc!r}); trying next")
    return None, "no readable checkpoint with smaller g^2 -> fresh start"


def main():
    Lx, Ly = parse_lattice(LATTICE)
    log("="*78)
    log(f"SU(2) (2+1)D  {LATTICE}   NONLOCAL MAGIC BOUNDS   chi={CHI}   mass={MASS}")
    log("  lower bound 4F from the Schmidt spectrum (exact, free)")
    log("  upper bound M_2 (total magic) -- exact FWHT while n <= "
        f"{EXACT_MAX_QUBITS} qubits")
    log("  observables: electric <E^2> and magnetic <B^2> densities")
    log("="*78)

    rows = []
    ed_rows = []
    # (g^2, psi) of the point handled just before in this run.  The warm-start
    # seed for each DMRG run is chosen by _warm_start_seed (fresh start at the
    # smallest g^2, otherwise the nearest smaller-g^2 checkpoint / previous point).
    previous = None
    g2_run = sorted(G2_VALUES)            # continuation goes upward in g^2
    if g2_run != list(G2_VALUES):
        log(f"  [warm-start] G2_VALUES reordered ascending: {g2_run}")
    json_file = _json_path()
    done = _done_g2(json_file) if RESUME else set()
    timed = []                            # per-point walltimes measured in this run
    stopped_early = False                 # deadline reached      -> exit EXIT_RESUBMIT
    seed_wait = False                     # --seed-chi not ready  -> exit EXIT_SEED_WAIT
    log(f"  output: {json_file}   checkpoints: {CKPT_DIR}/gs_{LATTICE}_m{_mtag()}_g*_chi{CHI}.pkl")
    if done:
        log(f"  [resume] g^2 already in {json_file}: {sorted(done)}")
    for g2 in g2_run:
        if any(abs(g2 - d) < 1e-12 for d in done):
            log(f"\n  g^2={g2:<7} [resume] already done -> skipped")
            previous = None               # next seed = this point's checkpoint on disk
            continue
        if DEADLINE is not None:
            # DMRG is checkpointed every sweep, so a point need not fit in one job:
            # start (or continue) it if about two sweeps still fit
            est, left = _point_seconds_estimate(timed), DEADLINE - time.time()
            need = 2.0*est/(MAX_SWEEPS + 1)
            if need > left:
                log(f"\n  [deadline] g^2={g2}: ~{need/3600:.2f} h needed for two sweeps, only "
                    f"{left/3600:.2f} h left -> stopping here (exit {EXIT_RESUBMIT}, "
                    f"resubmit to continue)")
                stopped_early = True
                break
        t0 = time.time()
        ckpt = _ckpt_path(LATTICE, g2, CHI)
        r = load_gs(ckpt, Lx, Ly, g2) if (LOAD_GS and not FORCE_RECOMPUTE) else None
        t_gs_before = 0.0                 # GS time spent in earlier jobs
        if r is not None:
            t_gs_before = float(r.get('walltime_gs', 0.0))
            log(f"\n  g^2={g2:<7} loaded checkpoint {ckpt}  eps={r['eps']:.10f}")
        else:
            partial = None if FORCE_RECOMPUTE else _load_partial(g2)
            psi0 = None
            if partial is not None:
                t_gs_before = float(partial.get('walltime_gs', 0.0))
                log(f"\n  g^2={g2:<7} [resume] continuing the interrupted DMRG run after "
                    f"sweep {partial['sweeps']}  ({_partial_path(g2)})")
            else:
                try:
                    psi0, why = _warm_start_seed(g2, previous)
                except _SeedNotReady as exc:
                    log(f"\n  g^2={g2:<7} [seed] {exc} does not exist yet -> exit "
                        f"{EXIT_SEED_WAIT} (retry when the chi={SEED_CHI} run has reached this g^2)")
                    seed_wait = True
                    break
                log(f"\n  g^2={g2:<7} [warm-start] {why}")
            try:
                r = run_plain_gs(Lx, Ly, g2, CHI, psi0=psi0, verbose=PROFILE_TIMINGS,
                                 resume=partial,
                                 on_sweep=_make_sweep_hook(g2, t_gs_before, t0))
            except _OutOfTime as exc:
                log(f"    [deadline] {exc} -> exit {EXIT_RESUBMIT}, resubmit to continue")
                stopped_early = True
                break
            psi0 = partial = None         # release the seed / resume MPS
            r['walltime_gs'] = t_gs_before + (time.time() - t0)
            if SAVE_GS:
                save_gs(ckpt, r)
            if os.path.exists(_partial_path(g2)):
                os.remove(_partial_path(g2))
        psi, qps = r['psi'], r['qps']
        previous = (g2, psi.copy()) if WARM_START_G2 else None
        bs = bipartite_scan(psi)
        mid = bs[len(bs)//2]
        n = int(sum(qps))
        m2, m2e, meth = np.nan, 0.0, "skipped"
        t_magic = 0.0
        if COMPUTE_MAGIC:
            _tm = time.perf_counter()
            As_magic = mps_arrays(psi, perm=r['perm'])
            if CAMPS_MODE == "magic":
                # SRE is Clifford-invariant: apply an entanglement-reducing
                # Clifford circuit, then read the SAME total M_2 off a lower-chi
                # (CHI_MAGIC) state.  This is the lever for larger lattices, where
                # the perfect-sampling cost grows with chi.
                As_magic, ci = camps_reduce(As_magic, qps, verbose=PROFILE_TIMINGS)
                log(f"    CAMPS[{CAMPS_MODE}] disentangle: chi "
                    f"{ci['chi_before']} -> {ci['chi_after']} (cap CHI_MAGIC="
                    f"{CHI_MAGIC})   {ci['n_gates']} Clifford gates, sum S "
                    f"{ci['S_before']:.3f} -> {ci['S_after']:.3f}")
            elif CAMPS_MODE == "gs":
                raise ValueError(
                    "CAMPS_MODE='gs' is the variational CAMPS ground state and "
                    "uses run_camps_gs() directly (different state basis, no "
                    "gauge-observable path); this scan supports 'off' and "
                    "'magic'.  Use 'magic' for cheaper large-lattice magic.")
            elif CAMPS_MODE != "off":
                raise ValueError(f"unknown CAMPS_MODE={CAMPS_MODE!r} "
                                 "(expected 'off' or 'magic')")
            m2, m2e, meth = total_M2(As_magic, qps)
            t_magic = time.perf_counter() - _tm
        dt = time.time()-t0
        if PROFILE_TIMINGS:
            log(f"    timing summary: magic={t_magic:.2f}s  total-point={dt:.2f}s")
        log(f"  g^2={g2:<7} eps={r['eps']:.10f}   ({dt:.1f}s)")
        log(f"    <E^2>={r['E2']:.6f}   <B^2>={r['B2']:.6f}   "
            f"Re<B_p>={r['ReBp']:.6f}")
        log(f"    {'bond':>5} {'chi':>5} {'S (bits)':>10} {'F':>12} {'4F':>12} {'F_Q':>12}")
        for d in bs:
            log(f"    {d['bond']:>5} {d['chi']:>5} {d['S']:>10.5f} {d['F']:>12.3e}"
                f" {d['4F']:>12.3e} {d['F_Q']:>12.3e}")
        don = None
        if COMPUTE_DONNELLY_EE:
            don = donnelly_ee_middle_cut(psi, Lx, Ly,
                                         max_sectors=DONNELLY_MAX_SECTORS)
            if don is None:
                log("    Donnelly EE: skipped (needs Lx>=2)")
            elif 'error' in don:
                log(f"    Donnelly EE: {don['error']} (bond {don['cut_bond']}); "
                    f"raise DONNELLY_MAX_SECTORS to compute")
            else:
                log(f"    Donnelly EE (vertical cut, bond {don['cut_bond']}, bits): "
                    f"S_total = {don['S_total']:.5f} = S_class {don['S_classical']:.5f}"
                    f" + S_quant {don['S_quantum']:.5f} + S_edge {don['S_edge']:.5f}"
                    f"   [S_standard={don['S_standard']:.5f}, {don['n_sectors']} "
                    f"sectors, p_norm={don['p_norm']:.4f}]")
                log(f"    (diagnostic) SU(2) irrep-resolved antiflatness at same "
                    f"cut: 4F_GI = {don['4F_GI']:.4e}   [gauge-invariant / 1D-style;"
                    f" the reported bound on the whole-state M2^NL is the "
                    f"whole-state 4F]")
        m2nl, nl_eta = np.nan, 0.0
        if COMPUTE_NL_MAGIC:
            nl = nonlocal_magic_reference(schmidt_values(psi, mid['bond']),
                                          max_rank=NL_MAGIC_MAX_RANK)
            m2nl, nl_eta = nl['M2_NL'], nl['eta']
            # the sandwich  4F <= M2^NL <= M2  must hold (up to sampling error on M2)
            flags = []
            if m2nl < mid['4F'] - 1e-6:
                flags.append("below 4F")
            if np.isfinite(m2) and m2nl > m2 + max(3.0*m2e, 1e-6):
                flags.append("above M2")
            tag = "" if not flags else "  *** BOUND VIOLATION: " + ", ".join(flags)
            eta_s = f", eta={nl_eta:.1e}" if nl_eta > 0 else ""
            log(f"    center bond: 4F = {mid['4F']:.4e}  <=  M2^NL = {m2nl:.4f}"
                f"  <=  M2 = {m2:.4f} +/- {m2e:.4f}"
                f"   [NL-ref: {nl['chi_kept']} Schmidt vals -> {nl['n_ref']} qubits"
                f"{eta_s}]{tag}")
            if VALIDATE_NL:
                sv_mid = schmidt_values(psi, mid['bond'])
                if NL_VALIDATE_METHOD == "riemannian":
                    vv = nonlocal_magic_minimize_riemannian(
                        sv_mid, keep=NL_VALIDATE_KEEP)
                    if vv is None:
                        log("      [NL validate] riemannian requested but pymanopt/"
                            "autograd unavailable -> falling back to L-BFGS")
                        vv = nonlocal_magic_minimize(sv_mid, keep=NL_VALIDATE_KEEP,
                                                     restarts=NL_VALIDATE_RESTARTS)
                    meth_tag = "riemannian(pymanopt)"
                else:
                    vv = nonlocal_magic_minimize(sv_mid, keep=NL_VALIDATE_KEEP,
                                                 restarts=NL_VALIDATE_RESTARTS)
                    meth_tag = "L-BFGS"
                if vv is not None:
                    gap = vv['M2_NL_ref'] - vv['M2_NL_min']   # >=0 if conjecture ok
                    verdict = ("ok" if abs(gap) < 1e-3 else
                               ("min<ref: conjecture questionable"
                                if gap > 1e-3 else "min>ref: not converged"))
                    log(f"      [NL validate] reference (conj.) M2^NL = "
                        f"{vv['M2_NL_ref']:.5f}  vs  {meth_tag} min = "
                        f"{vv['M2_NL_min']:.5f}   (gap {gap:+.2e}, keep="
                        f"{vv['chi_kept']}, eta={vv['eta']:.1e})  [{verdict}]")
        else:
            log(f"    center bond: 4F = {mid['4F']:.4e}   <=  M2^NL  <=  M2 = {m2:.4f}"
                f" +/- {m2e:.4f}   [{meth}]  ({n} qubits)")
        rows.append({'g2': g2, 'eps': r['eps'], 'S': mid['S'], 'F': mid['F'],
                     '4F': mid['4F'], 'F_Q': mid['F_Q'], '4F_Q': 4*mid['F_Q'],
                     'M2': m2, 'M2_err': m2e, 'M2_NL': m2nl, 'NL_eta': nl_eta,
                     'E2': r['E2'], 'B2': r['B2'],
                     'ReBp': r['ReBp'], 'chi11': creutz_11(r['ReBp']),
                     'F_sum': float(sum(d['F'] for d in bs)),
                     '4F_max': float(max(d['4F'] for d in bs)),
                     'S_don_total': (don.get('S_total', np.nan)
                                     if isinstance(don, dict) else np.nan),
                     'S_don_classical': (don.get('S_classical', np.nan)
                                         if isinstance(don, dict) else np.nan),
                     'S_don_quantum': (don.get('S_quantum', np.nan)
                                       if isinstance(don, dict) else np.nan),
                     'S_don_edge': (don.get('S_edge', np.nan)
                                    if isinstance(don, dict) else np.nan),
                     '4F_GI': (don.get('4F_GI', np.nan)
                               if isinstance(don, dict) else np.nan)})
        # walltime [s] of this g^2 point: ground state (DMRG + observables, also when
        # it was done by an earlier job and loaded here) + all measurements above
        walltime = (time.time() - t0) + t_gs_before
        rows[-1]['walltime'] = float(walltime)
        timed.append(walltime)
        _plot_and_dump([rows[-1]])        # write/refresh the JSON after EVERY point
        log(f"    walltime = {walltime:.1f} s  ({walltime/3600:.2f} h)  -> {json_file}")

        ed = run_ed(Lx, Ly, g2) if COMPUTE_ED else None
        ed_rows.append(ed)
        if ed is not None:
            log(f"    ED (exact): eps={ed['eps']:.10f}  <E^2>={ed['E2']:.5f}  "
                f"<B^2>={ed['B2']:.5f}  Re<B_p>={ed['ReBp']:.5f}  S={ed['S']:.4f}"
                f"  M2^NL={ed['M2_NL']:.4f}  M2={ed['M2']:.4f}")

    print ("WAITING FOR SEED:" if seed_wait else
           "STOPPED (deadline, resubmit):" if stopped_early else "COMPLETED:",
           datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S"))
    if rows:
        _plot_and_dump(rows)              # idempotent re-merge of this run's rows
    endTime = time.time()
    print ("Running time:", round(endTime - startTime, 2),  "seconds")
    # (rows were already written to the JSON after every g^2 point)

    log("\n" + "="*78)
    log(f"{'g^2':>7} {'eps':>15} {'<E^2>':>9} {'<B^2>':>9} {'S':>8}"
        f" {'Re<Bp>':>9} {'4F':>11} {'M2^NL':>9} {'M2':>9}")
    for x in rows:
        log(f"{x['g2']:>7} {x['eps']:>15.10f} {x['E2']:>9.5f} {x['B2']:>9.5f}"
            f" {x['S']:>8.4f} {x['ReBp']:>9.4f} {x['4F']:>11.3e}"
            f" {x.get('M2_NL', float('nan')):>9.4f} {x['M2']:>9.4f}")
    log("  (sandwich:  4F  <=  M2^NL  <=  M2 ;  M2^NL from the middle-cut Schmidt"
        " spectrum, Liu & Cui arXiv:2607.10714)")
    log("\n  chi(1,1) = -ln<plaquette> is the string tension at scale 1 (MONOTONIC).")
    log("  The roughening transition peaks in the susceptibilities above, not in")
    log("  chi(1,1).  Larger Creutz ratios need j_max>1/2 (see notes).")
    return EXIT_SEED_WAIT if seed_wait else (EXIT_RESUBMIT if stopped_early else 0)


def run_ed(Lx, Ly, g2):
    """Exact diagonalization of the SAME Hamiltonian for small lattices.  Returns
    the exact eps, <E^2>, <B^2>, Re<B_p>, S (middle cut) and, when n is small
    enough, the exact magic M_2 (FWHT) and nonlocal magic M_2^NL.  Returns None if
    the lattice is too large (n > ED_MAX_QUBITS or Hilbert space > ED_MAX_SIZE)."""
    sites, qps, coords, Nch = build_sites(Lx, Ly, verbose=False)
    n = int(sum(qps))
    if n > ED_MAX_QUBITS or (2.0**n)**2 > ED_MAX_SIZE:
        return None
    try:
        from tenpy.algorithms.exact_diag import ExactDiag
    except Exception:
        return None
    lat  = TeNPyLattice([1, 1], sites, bc='open', bc_MPS='finite')
    model = build_model(lat, Lx, Ly, g2, MASS, True)     # with penalty -> physical GS
    ED = ExactDiag(model, max_size=ED_MAX_SIZE)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization()
    _E0, gs = ED.groundstate()
    psi = ED.full_to_mps(gs)
    perm = [np.asarray(getattr(s, 'perm', np.arange(s.dim))) for s in sites]
    phys = build_model(lat, Lx, Ly, g2, MASS, False)
    eps = float(np.real(phys.H_MPO.expectation_value(psi))) / (Lx * Ly)
    obs = measure_gauge_observables(psi, lat, Lx, Ly)
    bs  = bipartite_scan(psi); mid = bs[len(bs) // 2]
    out = {'eps': eps, 'E2': obs['E2'], 'B2': obs['B2'], 'ReBp': obs['ReBp'],
           'S': mid['S'], 'M2': np.nan, 'M2_NL': np.nan}
    if COMPUTE_NL_MAGIC:
        out['M2_NL'] = nonlocal_magic_reference(
            schmidt_values(psi, mid['bond']), max_rank=NL_MAGIC_MAX_RANK)['M2_NL']
    if COMPUTE_MAGIC and n <= EXACT_MAX_QUBITS:
        Q = split_sites_to_qubits(mps_arrays(psi, perm=perm), qps)
        out['M2'] = float(m2_exact_fwht(mps_to_vector(Q), n))
    return out


def _ed_curve(ax, g, ed_rows, key, label="ED (exact)"):
    """Overlay the exact-diagonalization values for `key` as a dashed black line,
    skipping g^2 points where ED was unavailable."""
    xs = [gg for gg, e in zip(g, ed_rows) if e is not None and np.isfinite(e.get(key, np.nan))]
    ys = [e[key] for e in ed_rows if e is not None and np.isfinite(e.get(key, np.nan))]
    if xs:
        ax.semilogx(xs, ys, "k--", lw=1.6, marker="x", ms=6, mew=1.6,
                    zorder=5, label=label)


def _plot_observables(rows, ed_rows):
    """The four requested summary panels vs g^2, with dashed-black ED overlays
    (when available):  (1) gauge observables, (2) magic sandwich, (3) energy,
    (4) entanglement entropy."""
    g = [x['g2'] for x in rows]
    have_ed = any(e is not None for e in ed_rows)
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.6))
    a0, a1, a2, a3 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    # (1) gauge observables: <E^2>, <B^2>, average plaquette Re<B_p>
    a0.semilogx(g, [x['E2']   for x in rows], "--o", color="#1565c0", ms=7, lw=2,
                label=r"$\langle E^2\rangle$")
    a0.semilogx(g, [x['B2']   for x in rows], "--s", color="#c62828", ms=6, lw=2,
                mfc='none', label=r"$\langle B^2\rangle$")
    a0.semilogx(g, [x['ReBp'] for x in rows], "--^", color="#2e7d32", ms=7, lw=2,
                label=r"avg plaquette $\mathrm{Re}\langle B_p\rangle$")
    for k in ('E2', 'B2', 'ReBp'):
        _ed_curve(a0, g, ed_rows, k, label=None)
    a0.set_ylabel("gauge observables", fontsize=13)
    a0.set_title(r"(1)  $\langle E^2\rangle$, $\langle B^2\rangle$, avg plaquette")

    # (2) magic sandwich: 4F <= M2^NL <= M2
    a1.semilogx(g, [x['4F'] for x in rows], "--o", color="#bf360c", ms=7, lw=2,
                label=r"$4F$ (lower bound)")
    if np.isfinite([x.get('M2_NL', np.nan) for x in rows]).any():
        a1.semilogx(g, [x.get('M2_NL', np.nan) for x in rows], "--D",
                    color="#6a1b9a", ms=6, lw=2, label=r"$M_2^{\rm NL}$ (nonlocal)")
    if np.isfinite([x['M2'] for x in rows]).any():
        a1.errorbar(g, [x['M2'] for x in rows], yerr=[x['M2_err'] for x in rows],
                    fmt="--^", color="#1a237e", ms=7, lw=2, capsize=3,
                    label=r"$M_2$ (full SRE)")
        a1.fill_between(g, [x['4F'] for x in rows], [x['M2'] for x in rows],
                        color="#9fa8da", alpha=0.22)
    for k in ('4F', 'M2_NL', 'M2'):
        _ed_curve(a1, g, ed_rows, k, label=None)
    a1.set_ylabel("magic (bits)", fontsize=13); a1.set_ylim(bottom=0)
    a1.set_title(r"(2)  $4F \leq M_2^{\rm NL} \leq M_2$")

    # (3) ground-state energy density
    a2.semilogx(g, [x['eps'] for x in rows], "--o", color="#7b1fa2", ms=7, lw=2,
                label=r"DMRG $\varepsilon_b$")
    _ed_curve(a2, g, ed_rows, 'eps')
    a2.set_ylabel(r"$\varepsilon_b$", fontsize=14)
    a2.set_title("(3)  ground-state energy density")

    # (4) entanglement entropy (middle cut)
    a3.semilogx(g, [x['S'] for x in rows], "--o", color="#00695c", ms=7, lw=2,
                label=r"$S$ (center bond)")
    _ed_curve(a3, g, ed_rows, 'S')
    a3.set_ylabel(r"$S$ (bits)", fontsize=13)
    a3.set_title("(4)  entanglement entropy")

    for a in (a0, a1, a2, a3):
        a.set_xlabel(r"$g^2$", fontsize=14); a.grid(True, alpha=0.5)
        a.legend(fontsize=9)
    ttl = rf"SU(2) (2+1)D {LATTICE}, $b=0$, $\chi={CHI}$"
    if have_ed:
        ttl += "   (dashed black = exact diagonalization)"
    fig.suptitle(ttl, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"observables_{LATTICE}.pdf", bbox_inches="tight")
    fig.savefig(f"observables_{LATTICE}.png", bbox_inches="tight", dpi=140)
    log(f"\n  saved 4-panel summary -> observables_{LATTICE}.pdf (+ .png)")


def _plot_and_dump(rows):
    """Merge rows into nonlocal_magic_<lattice>_D<chi>_m<mass>.json: same list-of-
    dicts format and keys as before (+ 'walltime' in s), one row per g^2 (a new row
    replaces an existing one with the same g^2), sorted by g^2.  Written to a temp
    file and renamed, so a killed job never leaves a half-written JSON."""
    json_file = _json_path()
    new_g = {float(x['g2']) for x in rows}
    # several segment jobs (different g^2) can write the same file -> lock
    lockf = open(json_file + ".lock", "a")
    try:
        try:
            import fcntl
            fcntl.lockf(lockf, fcntl.LOCK_EX)
        except Exception:
            pass                          # no POSIX locks on this filesystem
        merged = [x for x in _read_rows(json_file) if float(x.get('g2', np.nan)) not in new_g]
        merged = sorted(merged + list(rows), key=lambda x: float(x['g2']))
        tmp = f"{json_file}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=1)
        os.replace(tmp, json_file)
    finally:
        lockf.close()                     # releases the lock


def _cli():
    """Override the module-level configuration from the command line.  Any option
    left unset keeps its default value defined at the top of the file, so the
    script still runs with no arguments."""
    import argparse
    p = argparse.ArgumentParser(
        description="SU(2) (2+1)D hardcore-gluon LGT: DMRG ground state, gauge "
                    "observables, and (nonlocal) magic on any lattice size.")
    # geometry
    p.add_argument("--lattice", default=None,
                   help="lattice as 'LxxLy', e.g. 6x6 (any size).")
    p.add_argument("--Lx", type=int, default=None, help="x extent (overrides --lattice)")
    p.add_argument("--Ly", type=int, default=None, help="y extent (overrides --lattice)")
    # core numerics
    p.add_argument("--chi", type=int, default=None, help="DMRG bond dimension")
    p.add_argument("--sweeps", type=int, default=None, help="max DMRG sweeps")
    p.add_argument("--mass", type=float, default=None, help="quark bare mass m")
    p.add_argument("--g2", type=float, nargs="+", default=None,
                   help="one or more g^2 values, e.g. --g2 0.5 1.0 2.0")
    # magic options
    p.add_argument("--camps-mode", choices=["off", "magic", "gs"], default=None)
    p.add_argument("--chi-magic", type=int, default=None,
                   help="bond dim for magic evaluation in CAMPS 'magic' mode")
    p.add_argument("--perfect-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="parallel sampler workers")
    p.add_argument("--exact-max-qubits", type=int, default=None)
    p.add_argument("--no-magic", action="store_true", help="skip total M_2")
    p.add_argument("--no-nl", action="store_true", help="skip nonlocal M_2^NL")
    p.add_argument("--no-validate-nl", action="store_true",
                   help="skip the local-unitary cross-check of M_2^NL")
    p.add_argument("--nl-validate-method", choices=["lbfgs", "riemannian"],
                   default=None, help="optimizer for the M_2^NL cross-check "
                   "(riemannian = pymanopt, as in arXiv:2607.10714)")
    p.add_argument("--nl-validate-keep", type=int, default=None,
                   help="Schmidt values kept in the M_2^NL cross-check "
                   "(raise toward the reported rank; riemannian scales better)")
    p.add_argument("--no-ed", action="store_true", help="skip exact-diag overlay")
    p.add_argument("--selftest", action="store_true",
                   help="check the 2x2 Hamiltonian against independent reference energies and exit")
    p.add_argument("--donnelly-ee", action="store_true",
                   help="compute the gauge-invariant (Donnelly) EE at the middle "
                   "vertical cut, sector-resolved by crossing-link SU(2) irreps")
    # checkpointing
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--no-load", action="store_true", help="do not load checkpoints")
    p.add_argument("--no-save", action="store_true", help="do not save checkpoints")
    # warm-start seeding
    p.add_argument("--cold-start-g2", type=float, default=None,
                   help="g^2 <= this starts fresh with no seed search (default "
                   "COLD_START_G2); pass a negative value to rely only on the "
                   "automatic 'no smaller-g^2 checkpoint -> fresh' rule")
    p.add_argument("--seed-g2", type=float, default=None,
                   help="manually force the checkpoint at this g^2 as the seed of "
                   "the first computed point (default: automatic search)")
    p.add_argument("--seed-pick", choices=["nearest", "smallest"], default=None,
                   help="which smaller-g^2 checkpoint to seed from (default nearest)")
    # batch / restart control (used by su2_scan.sh)
    p.add_argument("--deadline", type=float, default=None,
                   help="unix time by which to stop; a g^2 point that would not "
                   f"finish in time is not started (exit code {EXIT_RESUBMIT})")
    p.add_argument("--point-hours-guess", type=float, default=None,
                   help="expected hours per g^2 point, used until one has been timed")
    p.add_argument("--no-resume", action="store_true",
                   help="recompute g^2 points that already have a JSON row")
    p.add_argument("--force-recompute", action="store_true",
                   help="ignore existing ground-state checkpoints (re-run DMRG)")
    p.add_argument("--seed-chi", type=int, default=None,
                   help="segment mode: first g^2 of --g2 starts from the chi=SEED_CHI "
                   f"state at the same g^2 (exit {EXIT_SEED_WAIT} if it does not exist yet)")
    p.add_argument("--max-E-err", type=float, default=None,
                   help="DMRG max_E_err (default 1e-12; TeNPy applies it to |Delta E| of "
                   "the TOTAL energy when E<0)")
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest_hamiltonian() else 1)

    global LATTICE, CHI, MAX_SWEEPS, MASS, G2_VALUES, CAMPS_MODE, CHI_MAGIC
    global PERFECT_SAMPLES, PERFECT_WORKERS, EXACT_MAX_QUBITS
    global COMPUTE_MAGIC, COMPUTE_NL_MAGIC, VALIDATE_NL, COMPUTE_ED, COMPUTE_DONNELLY_EE
    global CKPT_DIR, LOAD_GS, SAVE_GS, NL_VALIDATE_METHOD, NL_VALIDATE_KEEP
    global COLD_START_G2, WARM_START_SEED_G2, WARM_START_SEED_PICK
    global DEADLINE, POINT_HOURS_GUESS, RESUME, FORCE_RECOMPUTE, SEED_CHI, MAX_E_ERR

    if a.Lx is not None or a.Ly is not None:
        if a.Lx is None or a.Ly is None:
            p.error("give both --Lx and --Ly, or use --lattice")
        LATTICE = f"{a.Lx}x{a.Ly}"
    elif a.lattice is not None:
        LATTICE = a.lattice
    parse_lattice(LATTICE)                      # validate early (and warn if odd)

    if a.chi              is not None: CHI = a.chi
    if a.sweeps           is not None: MAX_SWEEPS = a.sweeps
    if a.mass             is not None: MASS = a.mass
    if a.g2               is not None: G2_VALUES = list(a.g2)
    if a.camps_mode       is not None: CAMPS_MODE = a.camps_mode
    if a.chi_magic        is not None: CHI_MAGIC = a.chi_magic
    if a.perfect_samples  is not None: PERFECT_SAMPLES = a.perfect_samples
    if a.workers          is not None: PERFECT_WORKERS = a.workers
    if a.exact_max_qubits is not None: EXACT_MAX_QUBITS = a.exact_max_qubits
    if a.nl_validate_method is not None: NL_VALIDATE_METHOD = a.nl_validate_method
    if a.nl_validate_keep is not None: NL_VALIDATE_KEEP = a.nl_validate_keep
    if a.ckpt_dir         is not None: CKPT_DIR = a.ckpt_dir
    if a.no_magic:       COMPUTE_MAGIC = False
    if a.no_nl:          COMPUTE_NL_MAGIC = False
    if a.no_validate_nl: VALIDATE_NL = False
    if a.no_ed:          COMPUTE_ED = False
    if a.donnelly_ee:    COMPUTE_DONNELLY_EE = True
    if a.no_load:        LOAD_GS = False
    if a.no_save:        SAVE_GS = False
    if a.cold_start_g2    is not None:
        COLD_START_G2 = a.cold_start_g2 if a.cold_start_g2 >= 0 else None
    if a.seed_g2          is not None: WARM_START_SEED_G2 = a.seed_g2
    if a.seed_pick        is not None: WARM_START_SEED_PICK = a.seed_pick
    if a.deadline         is not None: DEADLINE = a.deadline
    if a.point_hours_guess is not None: POINT_HOURS_GUESS = a.point_hours_guess
    if a.no_resume:       RESUME = False
    if a.force_recompute: FORCE_RECOMPUTE = True
    if a.seed_chi         is not None: SEED_CHI = a.seed_chi
    if a.max_E_err        is not None: MAX_E_ERR = a.max_E_err


if __name__ == "__main__":
    _cli()
    sys.exit(main())     # 0 = all g^2 of this run done; 75 = resubmit; 76 = seed not ready
