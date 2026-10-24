# Non-stabilizerness in (2+1)D SU(2) lattice gauge theory

Code and data for R. G. Jha, J. I. Taher, M. Asaduzzaman, G. C. Toga, B. N. Bakalov, A. F. Kemper,
*Non-stabilizerness and entanglement in (2+1)-dimensional SU(2) lattice gauge theory using tensor networks*,
[arXiv:2609.28634](https://arxiv.org/abs/2609.28634) (v2).

## Contents
| | |
|---|---|
| `su2_magic.py` | DMRG (TeNPy) for the dressed-site, hard-core gluon SU(2) Hamiltonian at baryon number 0; gauge observables, entanglement (incl. classical/quantum/edge decomposition), M2, M2(ψ̃) ≈ M2^NL, anti-flatness F |
| `tests/test_ed.py` | 2×2 Hamiltonian self-test and DMRG vs. exact diagonalization |
| `data/` | all data of the paper: `nonlocal_magic_<L>x<L>_D<chi>_m<mass>.json` |
| `paper_plot.py`, `plot_ee4.py` | Figs. 2, 4, 5 and Fig. 3 (written to `figures/`) |

## Usage
```bash
pip install -r requirements.txt
python -m pytest tests -q              # ~15 s
python paper_plot.py && python plot_ee4.py
```
One scan (one lattice, χ, mass; one JSON row per g²):
```bash
python su2_magic.py --lattice 2x2 --chi 64 --mass 1 --g2 0.2 0.4 0.6 0.8 1.0 \
       --camps-mode off --no-validate-nl --no-ed          # add --no-magic to skip M2
```
Ground states are checkpointed in `gs_checkpoints/`; each g² starts from the nearest smaller-g² state.
The paper's 8×8, χ=128 runs take ~1 day per g² point on 2 cores.

## JSON keys (one row per g²)
`g2`, `eps` (energy per site), `E2`, `ReBp`, `B2` (gauge observables), `S` and `S_don_*` (entanglement entropy
at the central cut and its decomposition, bits), `F`, `4F` (anti-flatness; lower bound = −log2(1−4F)),
`M2_NL` (= M2(ψ̃), SRE of the computational-basis Schmidt representative), `M2`, `M2_err` (full SRE, 4×4 only),
`walltime` (s).

## Note
The on-site basis is fixed by an eigensolver up to rotations inside degenerate pairs and phases, which depend on
the LAPACK build. M2 depends on this choice; all other quantities do not.

## License
MIT, see `LICENSE`.
