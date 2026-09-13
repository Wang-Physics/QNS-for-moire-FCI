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


@lru_cache(None)
def first_bz_c6_grid():
    """C6-closed set of physical Fourier vectors in the closed first BZ.

    A 27-element momentum-class mesh cannot itself be a closed point set on
    the boundary: a corner is fixed only *modulo* a reciprocal vector.  For a
    continuum observable such as S(q), the rotated boundary vectors are
    distinct physical Fourier modes.  Complete the plotted class
    representatives under C3 (inversion then also gives C6) and retain every
    distinct vector.  The present 27-cell cluster gives 37 points.
    """
    source = first_bz_momenta()
    candidates = []
    for turn in range(3):
        angle = 2.0 * np.pi * turn / 3.0
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        candidates.extend(source @ rotation.T)
    unique = []
    for point in candidates:
        point = np.where(np.abs(point) < 1.0e-15, 0.0, point)
        if not any(np.linalg.norm(point - other) < 1.0e-12 for other in unique):
            unique.append(point)
    points = np.asarray(unique)
    angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    order = np.lexsort((angles, np.round(np.linalg.norm(points, axis=1), 13)))
    points = points[order]
    model, qns_inputs = inputs()
    if points.shape != (37, 2):
        raise RuntimeError("the C6-closed 27-cell first-BZ grid must have 37 points")
    if not all(model.inside_hexagon(point) for point in points):
        raise ValueError("completed Fourier grid contains a point outside the first BZ")
    reciprocal_coordinates = (
        points @ qns_inputs["supercell_lattice"] / (2.0 * np.pi)
    )
    if not np.allclose(reciprocal_coordinates, np.rint(reciprocal_coordinates), atol=1e-12):
        raise ValueError("completed Fourier grid is incompatible with the supercell")
    return points


def momentum_class_indices(points):
    """Map physical first-BZ vectors to the 27 reciprocal-lattice classes."""
    points = np.asarray(points, dtype=float)
    _, qns_inputs = inputs()
    fractional = points @ np.linalg.inv(qns_inputs["primitive_reciprocal"])
    numerators = np.rint(fractional * cluster().determinant).astype(np.int64)
    if not np.allclose(
        fractional, numerators / cluster().determinant, rtol=0.0, atol=1e-12
    ):
        raise ValueError("point is not on the 27-cell momentum mesh")
    numerators %= cluster().determinant
    indices = cluster().residue_lookup[numerators[:, 0], numerators[:, 1]]
    if np.any(indices < 0):
        raise ValueError("point does not belong to a 27-cell momentum class")
    return indices
