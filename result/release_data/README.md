# Compact release data

These files are the minimal numerical inputs used by the publication plotting scripts:

- `traces/`: four 120-update baselines, the stopped 116-update internal-C3 trace,
  and all three 120-update outer-`C3` sectors at both fillings;
- `diagnostics/nu*outer_c3_p{0,1,2}.{npz,json}`: complete-ratio 1-RDM,
  band weights, `n_1(k)`, full `S(q)`, and folded density maps measured from
  all six equilibrated final training ensembles;
- the validation-summary JSON files are retained as historical audit artifacts
  but are not inputs to the current training-tail figures or energy comparison.

Each outer-sector diagnostic uses 4,128 configurations and 16 jointly batched
auxiliary replacements per configuration. Every replacement recomputes the
complete projected wavefunction. Raw coordinates, checkpoints, and optimizer
states are intentionally omitted from Git.
