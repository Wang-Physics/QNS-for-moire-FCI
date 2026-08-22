# Result artifacts

This directory separates publication artifacts from large local numerical output.

Tracked:

- `AI_for_Physics.tex` and `neural_bloch_part2.tex`;
- the compiled paper-style PDF;
- `figures/` in PDF and PNG formats;
- `release_data/`, the compact input needed to redraw Figs. 6–7;
- small audit summaries such as `one_rdm_hamiltonian_audit.json`.

Not tracked:

- `data/` training checkpoints, walkers, local-energy samples, Lanczos vectors, and twist-state caches, except for explicitly whitelisted small ED regression fixtures;
- LaTeX auxiliary files and local build logs.

The ignored raw data remain local. Reproducing them requires running the ED and VMC entry points described in the project README.
