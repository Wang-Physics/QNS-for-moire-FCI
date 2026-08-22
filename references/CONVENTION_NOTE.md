# Continuum convention selected for the active calculation

The target paper's compact Appendix Eq. (12) writes a layer-phase/tunneling
convention that is easy to combine inconsistently with its Eq. (11).  For the
actual plane-wave matrix, this implementation follows the explicit formulas in
Reddy et al., Phys. Rev. B 108, 085117 (2023), Supplemental Eqs. (3)--(6):

- `phi_2 = -phi_1 = phi`;
- the upper off-diagonal block has transfers `0`, `+g2`, and `+g3`.

This choice reproduces the reference Fig. 3 result that the first and second
bands at `theta=3 degrees` both have `C=-1`, up to the sign convention for the
chosen valley.  The discarded mixed convention and all of its outputs are
preserved under the run's `invalid_convention_run_001/` directory.
