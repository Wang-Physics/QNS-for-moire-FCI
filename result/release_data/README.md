# Compact release data

These files are the minimal numerical inputs used by the publication plotting scripts:

- `traces/`: four 120-update training traces for `nu=1/3,2/3`, with and without `M`;
- `validation_summary.json`: low-cost selection and precise-validation energies;
- `diagnostics/`: compressed band weights, `n_1(k)`, full `S(q)`, and folded density maps.

The full-`M` diagnostics use the independent 1,024-walker validation ensembles. The no-`M` diagnostics use independently re-equilibrated 4,128-walker precise-validation ensembles. Raw coordinates, checkpoints, and optimizer states are intentionally omitted from Git.
