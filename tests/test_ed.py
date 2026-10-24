"""Small-lattice tests:  python -m pytest tests -q   (a few minutes)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import su2_magic as su

def test_hamiltonian_selftest():
    """2x2 spectrum vs. an independent link-basis construction (reference energies)."""
    assert su.selftest_hamiltonian(verbose=False)

def test_dmrg_vs_ed_2x2():
    """DMRG ground state on 2x2 (12 qubits) reproduces exact diagonalization."""
    su.MAX_SWEEPS, su.PROFILE_TIMINGS = 20, False
    for g2 in (0.6, 1.0, 4.0):
        ed = su.run_ed(2, 2, g2)
        r = su.run_plain_gs(2, 2, g2, 32, verbose=False)
        mid = su.bipartite_scan(r['psi'])[1]
        nl = su.nonlocal_magic_reference(su.schmidt_values(r['psi'], mid['bond']))['M2_NL']
        assert abs(r['eps'] - ed['eps']) < 1e-8
        for k in ('E2', 'ReBp'):
            assert abs(r[k] - ed[k]) < 1e-6, (g2, k)
        assert abs(mid['S'] - ed['S']) < 1e-6
        assert abs(nl - ed['M2_NL']) < 1e-6
        assert -np.log2(1 - mid['4F']) <= nl + 1e-12          # lower bound
        assert nl <= ed['M2'] + 1e-9                          # M2(psi~) <= M2 (CB conjecture)
