# Paper notes: arXiv:2503.13585v3 and continuum ED reference

## Target paper

Di Luo, Timothy Zaklama, and Liang Fu, "Solving fractional electron states in
twisted MoTe2 with deep neural network," arXiv:2503.13585v3 (7 April 2025).
Local PDF: `2503.13585v3.pdf`.

- **PAPER, p.1, Eq. (1):** interacting two-layer continuum Hamiltonian.
- **PAPER, p.2, paragraph below Eq. (1):** periodic-image Coulomb interaction
  and parameters `(V,w,phi,m*)=(11.2 meV,13.3 meV,-91 deg,0.62 me)`.
- **PAPER, p.3, Fig. 2 and Results paragraph:** multiband 1--5 ED uses
  `theta=3 deg`, `nu=2/3`, `3 x 3`, and `epsilon=5,10`.
- **PAPER, p.3, Fig. 3:** `nu=1/3` comparison uses 1-band ED and a 27-cell
  cluster, not multiband ED.
- **PAPER, p.8, Eqs. (11)--(14):** gauge-transformed one-body Hamiltonian,
  moire potentials, tunneling, moire period, and Coulomb interaction.
- **PAPER, p.9, Eq. (15):** multiband Bloch-basis projected Hamiltonian and
  translation-sector ED description.
- **PAPER, p.9 and Fig. 5(a,b):** at `theta=2.6 deg`, `epsilon=5`, `nu=1/3`
  has three low states at Gamma/K/K' and a structure-factor Bragg peak,
  interpreted by the authors as CDW order.

## Primary continuum/ED reference

A. P. Reddy et al., Phys. Rev. B 108, 085117 (2023), arXiv:2304.12261.
Local PDF: `2304.12261.pdf`.

- **PAPER, main p.3, Table I:** `a0=3.52 Angstrom`, and the same MoTe2
  potential, phase, and effective mass (the sign of `w` is removable by a
  global layer rephasing).
- **PAPER, Supplemental pp.2--3, Eqs. (3)--(9):** explicit plane-wave matrix
  elements and projected Coulomb form factors.
- **PAPER, Supplemental p.3:** reciprocal lattice is truncated by radial
  shells; `N=30` was used and described as far beyond numerical necessity.
- **PAPER, Supplemental p.4:** `q=0` is removed, corresponding to the uniform
  neutralizing-background convention.
- **PAPER, Supplemental p.4:** Fock-bit representation and block
  diagonalization by particle number, spin, and crystal momentum.
- **PAPER, Supplemental p.8, Eqs. (19),(20):** Berry-curvature nonuniformity
  and trace-condition definitions.

## Derived extension used here

The active smoke run combines the target paper's multiband `3 x 3`,
`theta=3 deg`, `epsilon=5` setting with `nu=1/3`.  This exact combination is
not a published data point.  It is labeled **DERIVED**, and any agreement or
disagreement is restricted to this finite cluster and cutoff.
