# Compact release data

These files are the minimal numerical inputs used by the publication plotting scripts:

- `traces/`: seed-260904 full-`M` and v3 no-`M`+$M_S$ baselines at both
  fillings. Each v3.2 outer-`C3`+$M_S$ trace uses the same initialization seed
  across sectors and contains 20 independent-parameter updates evaluated on a
  shared mixture-sampled pool followed by 100 private-chain updates, with
  `stage` and `stage_step` recording that boundary;
- `diagnostics/nu*outer_c3_p{0,1,2}.{npz,json}`: complete-ratio 1-RDM,
  band weights, `n_1(k)`, full `S(q)`, and folded density maps measured from
  all six equilibrated final training ensembles;
- `summary.json` records the exact final-ten values used by the report, and
  `report_validation.json` records source-lineage and figure-integrity checks;
- historical validation-summary JSON files are retained as audit artifacts but
  are not inputs to the current training-tail figures or energy comparison.

Each outer-sector diagnostic uses 4,128 configurations and 16 jointly batched
auxiliary replacements per configuration. Every replacement recomputes the
complete projected wavefunction. Raw coordinates, checkpoints, and optimizer
states are intentionally omitted from Git.

`v5/` is the self-contained input bundle for Figs. 6--11 of report v5.
Its 27-cell diagnostics store direct layer-summed `G=0` plane-wave `n(k)` and
full `S(q)` at the exact 27 displayed first-BZ vectors. `v5/manifest.json`
records the source path, byte count, and SHA-256 digest of every bundled file.
