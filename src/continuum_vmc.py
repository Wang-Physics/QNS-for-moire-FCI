"""Continuous-space local energy and Metropolis sampler for neural Bloch VMC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from .continuum import ContinuumModel, reciprocal_shells
from .multiband_ed import COULOMB_MEV_NM
from .neural_bloch import ManyBodyNeuralBloch


@dataclass
class WalkerState:
    positions: torch.Tensor
    layers: torch.Tensor
    logpsi: torch.Tensor


class ContinuumTorusHamiltonian:
    """The paper continuum Hamiltonian on a rectangular n1 x n2 torus.

    The q=0 Coulomb component is removed, exactly as in the multiband ED
    benchmark.  Kinetic derivatives are evaluated by a symmetric finite
    difference of the complete complex wavefunction, so neural coordinate
    backflow and orbital transformations are both included.
    """

    def __init__(
        self,
        model: ContinuumModel,
        n1: int,
        n2: int,
        dielectric: float,
        reciprocal_shell_count: int,
        finite_difference_fraction: float = 1.5e-3,
    ) -> None:
        self.model = model
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.dielectric = float(dielectric)
        dtype = torch.get_default_dtype()
        reciprocal = np.stack([model.b1, model.b2])
        primitive_lattice = 2.0 * np.pi * np.linalg.inv(reciprocal)
        supercell_lattice = primitive_lattice @ np.diag([n1, n2])
        self.primitive_lattice = primitive_lattice
        self.supercell_lattice = supercell_lattice
        self.inverse_supercell = np.linalg.inv(supercell_lattice)
        self.area = n1 * n2 * model.cell_area
        self.finite_difference_nm = finite_difference_fraction * model.a_m
        self.supercell_tensor = torch.as_tensor(supercell_lattice, dtype=dtype)
        self.inverse_supercell_tensor = torch.as_tensor(
            self.inverse_supercell, dtype=dtype
        )
        self.kappas = torch.as_tensor(model.kappas, dtype=dtype)
        self.q_vectors, self.q_coefficients = self._coulomb_modes(
            reciprocal_shell_count
        )

    def _coulomb_modes(self, shell_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = []
        coefficients = []
        for du in range(-(self.n1 // 2), self.n1 // 2 + 1):
            for dv in range(-(self.n2 // 2), self.n2 // 2 + 1):
                qmesh = du * self.model.b1 / self.n1 + dv * self.model.b2 / self.n2
                for gm, gn in reciprocal_shells(shell_count):
                    q = qmesh + gm * self.model.b1 + gn * self.model.b2
                    norm = float(np.linalg.norm(q))
                    if norm < 1.0e-13:
                        continue
                    vectors.append(q)
                    coefficients.append(
                        2.0 * np.pi * COULOMB_MEV_NM
                        / (self.dielectric * norm * self.area)
                    )
        dtype = torch.get_default_dtype()
        return (
            torch.as_tensor(np.asarray(vectors), dtype=dtype),
            torch.as_tensor(np.asarray(coefficients), dtype=dtype),
        )

    def cartesian(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ac,bnc->bna", self.supercell_tensor, positions)

    def wrap(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.remainder(positions, 1.0)

    def coordinate_shift(self, axis: int, amount_nm: float) -> torch.Tensor:
        cartesian = torch.zeros(2, dtype=self.supercell_tensor.dtype)
        cartesian[axis] = amount_nm
        return torch.einsum("ac,c->a", self.inverse_supercell_tensor, cartesian)

    def diagonal_potential(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        cartesian = self.cartesian(positions)
        phase = np.deg2rad(self.model.params.phase_deg)
        layer_phase = torch.where(
            layers == 0,
            torch.as_tensor(-phase, dtype=cartesian.dtype),
            torch.as_tensor(phase, dtype=cartesian.dtype),
        )
        directions = torch.stack(
            [
                torch.as_tensor(self.model.b1, dtype=cartesian.dtype),
                torch.as_tensor(self.model.b2, dtype=cartesian.dtype),
                -torch.as_tensor(self.model.b1 + self.model.b2, dtype=cartesian.dtype),
            ]
        )
        arguments = torch.einsum("bna,ga->bng", cartesian, directions)
        moire = -2.0 * self.model.params.potential_meV * torch.sum(
            torch.cos(arguments + layer_phase[..., None]), dim=(1, 2)
        )

        displacement = cartesian[:, :, None, :] - cartesian[:, None, :, :]
        qphase = torch.einsum("bija,qa->bijq", displacement, self.q_vectors)
        pair_potential = torch.einsum(
            "q,bijq->bij", self.q_coefficients, torch.cos(qphase)
        )
        upper = torch.triu(
            torch.ones(
                layers.shape[1], layers.shape[1],
                dtype=cartesian.dtype, device=cartesian.device,
            ),
            diagonal=1,
        )
        coulomb = torch.sum(pair_potential * upper, dim=(1, 2))
        return moire + coulomb

    def tunneling_local_energy(
        self,
        wavefunction: ManyBodyNeuralBloch,
        positions: torch.Tensor,
        layers: torch.Tensor,
        base_logpsi: torch.Tensor,
    ) -> torch.Tensor:
        batch, particles = layers.shape
        flipped = layers[:, None, :].expand(-1, particles, -1).clone()
        particle_index = torch.arange(particles, device=layers.device)
        flipped[:, particle_index, particle_index] = 1 - flipped[
            :, particle_index, particle_index
        ]
        positions_many = positions[:, None].expand(-1, particles, -1, -1)
        logs = wavefunction(
            positions_many.reshape(batch * particles, particles, 2),
            flipped.reshape(batch * particles, particles),
        ).reshape(batch, particles)
        ratio = torch.exp(logs - base_logpsi[:, None])
        cartesian = self.cartesian(positions)
        b2 = torch.as_tensor(self.model.b2, dtype=cartesian.dtype)
        b12 = torch.as_tensor(self.model.b1 + self.model.b2, dtype=cartesian.dtype)
        t01 = self.model.params.tunneling_meV * (
            1.0
            + torch.exp(1j * torch.einsum("bna,a->bn", cartesian, b2))
            + torch.exp(1j * torch.einsum("bna,a->bn", cartesian, b12))
        )
        matrix_element = torch.where(layers == 0, t01, torch.conj(t01))
        return torch.sum(matrix_element * ratio, dim=1)

    def kinetic_local_energy(
        self,
        wavefunction: ManyBodyNeuralBloch,
        positions: torch.Tensor,
        layers: torch.Tensor,
        base_logpsi: torch.Tensor,
    ) -> torch.Tensor:
        batch, particles = layers.shape
        h = self.finite_difference_nm
        result = torch.zeros(batch, dtype=torch.complex128, device=positions.device)
        kappa = self.kappas[layers]
        for particle, axis in product(range(particles), range(2)):
            delta = self.coordinate_shift(axis, h).to(positions.device)
            plus = positions.clone()
            minus = positions.clone()
            plus[:, particle] = self.wrap(plus[:, particle] + delta)
            minus[:, particle] = self.wrap(minus[:, particle] - delta)
            plus_log = wavefunction(plus, layers)
            minus_log = wavefunction(minus, layers)
            plus_ratio = torch.exp(plus_log - base_logpsi)
            minus_ratio = torch.exp(minus_log - base_logpsi)
            first_ratio = (plus_ratio - minus_ratio) / (2.0 * h)
            second_ratio = (plus_ratio + minus_ratio - 2.0) / (h * h)
            result = result - self.model.alpha * second_ratio
            result = result + 2j * self.model.alpha * kappa[:, particle, axis] * first_ratio
        result = result + self.model.alpha * torch.sum(kappa * kappa, dim=(1, 2))
        return result

    @torch.no_grad()
    def local_energy(
        self,
        wavefunction: ManyBodyNeuralBloch,
        positions: torch.Tensor,
        layers: torch.Tensor,
    ) -> torch.Tensor:
        base_logpsi = wavefunction(positions, layers)
        kinetic = self.kinetic_local_energy(
            wavefunction, positions, layers, base_logpsi
        )
        diagonal = self.diagonal_potential(positions, layers)
        tunneling = self.tunneling_local_energy(
            wavefunction, positions, layers, base_logpsi
        )
        return kinetic + diagonal + tunneling


class ContinuousMetropolis:
    def __init__(
        self,
        wavefunction: ManyBodyNeuralBloch,
        hamiltonian: ContinuumTorusHamiltonian,
        walkers: int,
        position_step_fraction: float,
        seed: int,
        device: torch.device,
        wavefunction_batch_size: int = 256,
    ) -> None:
        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.walkers = int(walkers)
        self.position_step_fraction = float(position_step_fraction)
        self.wavefunction_batch_size = int(wavefunction_batch_size)
        if self.wavefunction_batch_size < 1:
            raise ValueError("wavefunction_batch_size must be positive")
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        positions = torch.rand(
            walkers, wavefunction.n_particles, 2,
            generator=self.generator, dtype=torch.get_default_dtype(), device=device,
        )
        layers = torch.randint(
            0, 2, (walkers, wavefunction.n_particles),
            generator=self.generator, dtype=torch.long, device=device,
        )
        with torch.no_grad():
            logpsi = self._evaluate(positions, layers)
        self.state = WalkerState(positions, layers, logpsi)

    def _evaluate(
        self, positions: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        values = []
        for start in range(0, len(positions), self.wavefunction_batch_size):
            stop = start + self.wavefunction_batch_size
            values.append(self.wavefunction(positions[start:stop], layers[start:stop]))
        return torch.cat(values)

    @torch.no_grad()
    def sweep(self, sweeps: int = 1) -> dict[str, float]:
        accepted_position = 0
        accepted_layer = 0
        attempted = self.walkers * sweeps
        for _ in range(sweeps):
            particle = torch.randint(
                self.wavefunction.n_particles,
                (self.walkers,), generator=self.generator,
                device=self.state.positions.device,
            )
            rows = torch.arange(self.walkers, device=self.state.positions.device)
            proposal = self.state.positions.clone()
            proposal[rows, particle] = self.hamiltonian.wrap(
                proposal[rows, particle]
                + self.position_step_fraction
                * torch.randn(
                    self.walkers, 2, generator=self.generator,
                    dtype=proposal.dtype, device=proposal.device,
                )
            )
            proposal_log = self._evaluate(proposal, self.state.layers)
            probability = torch.exp(
                2.0 * (proposal_log.real - self.state.logpsi.real)
            ).clamp(max=1.0)
            accept = torch.rand(
                self.walkers, generator=self.generator,
                dtype=proposal.dtype, device=proposal.device,
            ) < probability
            self.state.positions[accept] = proposal[accept]
            self.state.logpsi[accept] = proposal_log[accept]
            accepted_position += int(accept.sum())

            layer_proposal = self.state.layers.clone()
            layer_proposal[rows, particle] = 1 - layer_proposal[rows, particle]
            layer_log = self._evaluate(self.state.positions, layer_proposal)
            layer_probability = torch.exp(
                2.0 * (layer_log.real - self.state.logpsi.real)
            ).clamp(max=1.0)
            layer_accept = torch.rand(
                self.walkers, generator=self.generator,
                dtype=proposal.dtype, device=proposal.device,
            ) < layer_probability
            self.state.layers[layer_accept] = layer_proposal[layer_accept]
            self.state.logpsi[layer_accept] = layer_log[layer_accept]
            accepted_layer += int(layer_accept.sum())
        return {
            "position_acceptance": accepted_position / max(attempted, 1),
            "layer_acceptance": accepted_layer / max(attempted, 1),
        }

    @torch.no_grad()
    def refresh(self) -> None:
        self.state.logpsi = self._evaluate(
            self.state.positions, self.state.layers
        )
