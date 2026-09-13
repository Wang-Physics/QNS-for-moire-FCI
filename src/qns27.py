"""Exact 27-cell geometry and complete Gamma minor lists for continuum QNS."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .c6_ed import TriangularSupercell, cluster_bloch_states
from .continuum import ContinuumModel, ContinuumParameters, reciprocal_shells
from .group_sparse_ed import all_group_sector_bases
from .multiband_ed import COULOMB_MEV_NM

DATA = Path(__file__).resolve().parents[1]/'result/data'


@lru_cache(None)
def cluster():
    return TriangularSupercell.from_matrix(np.array([[3,-3],[3,6]]))


@lru_cache(None)
def gamma_indices(particles):
    if particles not in (9,18):
        raise ValueError('27 cells require 9 or 18 particles in this experiment')
    c=cluster()
    basis=all_group_sector_bases(c.numerators,27,c.residue_lookup,1,particles)[0]
    occupied=((basis[:,None] >> np.arange(27,dtype=np.uint64)) & 1).astype(bool)
    result=np.broadcast_to(np.arange(27),occupied.shape)[occupied].reshape(-1,particles)
    assert len(result)==173613
    assert np.all(c.numerators[result].sum(axis=1)%27==0)
    return result


@lru_cache(None)
def inputs():
    model=ContinuumModel(ContinuumParameters(theta_deg=2.6,plane_wave_shells=8))
    c=cluster(); cache=DATA/'qns27_bloch_inputs.npz'
    if not cache.exists():
        energies,vectors,_=cluster_bloch_states(model,c,1)
        # Coefficients were solved at unwrapped fractions, NOT displayed BZ k.
        np.savez_compressed(cache,energies=energies,vectors=vectors,matrix=c.matrix,theta=2.6)
    with np.load(cache) as z:
        assert np.array_equal(z['matrix'],c.matrix) and float(z['theta'])==2.6
        energies=z['energies'][:,0]; coefficients=z['vectors'][:,:,0].reshape(27,2,model.n_g)
    reciprocal=np.stack([model.b1,model.b2]); lattice=2*np.pi*np.linalg.inv(reciprocal)@c.matrix
    q=[]
    for f in c.centered_fractions:
        for m,n in reciprocal_shells(8):
            v=(f+np.array([m,n]))@reciprocal
            if np.linalg.norm(v)>1e-13: q.append(v)
    q=np.asarray(q)
    coeff=2*np.pi*COULOMB_MEV_NM/(5*np.linalg.norm(q,axis=1)*27*model.cell_area)
    return model,dict(supercell_lattice=lattice,primitive_reciprocal=reciprocal,
                      momentum_fractions=c.fractions,momenta=c.fractions@reciprocal,
                      g_vectors=model.g_vectors,bloch_coefficients=coefficients,
                      one_body_energies=energies,q_vectors=q,q_coefficients=coeff)


@lru_cache(None)
def first_bz_momenta():
    """Authoritative physical coordinates of the 27 dots drawn in the BZ.

    Boundary points have two equal-norm representatives separated by a
    primitive reciprocal vector.  Reading the publication mesh makes that
    otherwise ambiguous choice explicit and guarantees that observables are
    evaluated at the point that is actually displayed.
    """
    path = DATA / "fig5_27cell_observables.json"
    points = np.asarray(json.loads(path.read_text())["k_points_hex"], dtype=float)
    model, qns_inputs = inputs()
    if points.shape != (cluster().determinant, 2):
        raise ValueError("the publication BZ mesh must contain 27 two-vectors")
    fractional = points @ np.linalg.inv(qns_inputs["primitive_reciprocal"])
    reciprocal_shifts = np.asarray(cluster().fractions) - fractional
    if not np.allclose(reciprocal_shifts, np.rint(reciprocal_shifts), atol=1e-12):
        raise ValueError("publication BZ points do not match the 27 momentum classes")
    if not all(model.inside_hexagon(point) for point in points):
        raise ValueError("publication momentum lies outside the first BZ")
    return points
