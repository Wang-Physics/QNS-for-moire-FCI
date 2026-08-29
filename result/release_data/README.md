# Compact release data

These files are the minimal numerical inputs used by the publication plotting scripts:

- `traces/`: four 120-update baselines, the stopped 116-update internal-C3 trace,
  and all three 120-update outer-`C3` sectors at both fillings;
- `validation_summary.json`, `outer_c3_validation_summary.json`, and
  `outer_c3_nu2of3_validation_summary.json`: low-cost selection and precise energies;
- `diagnostics/`: compressed band weights, `n_1(k)`, full `S(q)`, and folded
  density maps, including the independently selected outer-projected states.

The full-`M` diagnostics use independent 1,024-walker validation ensembles.
The no-`M`, C3-QNS, and selected outer-`C3` diagnostics use independently
re-equilibrated 4,128-walker precise-validation ensembles. Raw coordinates,
checkpoints, and optimizer states are intentionally omitted from Git.
