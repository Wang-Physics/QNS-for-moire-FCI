# Validated plan: tMoTe2 Fig. 2 multiband ED

## Scientific objective

Validate the continuum and Coulomb-projection code against Fig. 2 of arXiv:2503.13585v3 before starting the QNS optimization.

## Locked benchmark

- `theta = 3 degrees`, dielectric constant `epsilon = 5`
- `3 x 3` moire cluster, nine cells
- filling `nu = 2/3`, six fully polarized holes
- continuum parameters `(V, w, phi, m*) = (11.2 meV, 13.3 meV, -91 degrees, 0.62 me)`
- periodic unscreened 2D Coulomb interaction with the neutralizing `q = 0` term removed
- 1-, 2-, 3-, and 5-band projections

## Acceptance conditions

- energy per hole agrees with digitized Fig. 2 values within `0.10 meV`;
- pair interaction is Hermitian within `1e-12`;
- 1--3 band calculations scan all nine total-momentum sectors;
- 5-band calculation solves all three point-group-inequivalent sectors;
- Lanczos residuals are much smaller than the digitization uncertainty;
- nested-band ground-state overlaps and `sum_k n(k) = 6` are recorded;
- figures use a restrained, color-blind-safe Nature/Science-style design.

## Outcome

All conditions passed. The next experiment may use these exact states, energies, overlaps, and momentum occupations as fixed QNS targets. No FCI phase claim is inferred from this finite cluster alone.
