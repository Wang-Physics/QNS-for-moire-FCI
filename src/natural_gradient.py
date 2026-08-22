"""Matrix-free stochastic reconfiguration for complex neural wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import functional_call, jvp, vjp


@dataclass(frozen=True)
class NaturalGradientDiagnostics:
    force_norm: float
    direction_norm: float
    residual_norm: float
    cg_iterations: int
    direction_was_clipped: bool


def _parameter_items(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in module.named_parameters()
            if parameter.requires_grad]


def _flatten(values: list[torch.Tensor], items) -> torch.Tensor:
    pieces = []
    for value, (_, parameter) in zip(values, items):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if value is None else value.reshape(-1)
        )
    return torch.cat(pieces)


def _unflatten(vector: torch.Tensor, items) -> dict[str, torch.Tensor]:
    output = {}
    offset = 0
    for name, parameter in items:
        count = parameter.numel()
        output[name] = vector[offset:offset + count].reshape_as(parameter)
        offset += count
    if offset != vector.numel():
        raise ValueError("natural-gradient vector has the wrong size")
    return output


def energy_force(
    wavefunction: torch.nn.Module,
    positions: torch.Tensor,
    layers: torch.Tensor,
    local_energy: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Return the real-parameter VMC energy gradient."""
    items = _parameter_items(wavefunction)
    parameters = [parameter for _, parameter in items]
    centered = local_energy - local_energy.mean()
    force = torch.zeros(
        sum(parameter.numel() for parameter in parameters),
        dtype=parameters[0].dtype,
        device=parameters[0].device,
    )
    total = len(positions)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        logpsi = wavefunction(positions[start:stop], layers[start:stop])
        loss = 2.0 * torch.real(
            torch.sum(centered[start:stop] * torch.conj(logpsi)) / total
        )
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=False
        )
        force = force + _flatten(list(gradients), items)
    return force.detach()


def fisher_vector_product(
    wavefunction: torch.nn.Module,
    positions: torch.Tensor,
    layers: torch.Tensor,
    vector: torch.Tensor,
    damping: float,
    chunk_size: int,
) -> torch.Tensor:
    """Apply the centered quantum-geometric tensor plus diagonal damping."""
    items = _parameter_items(wavefunction)
    parameter_dict = {name: parameter for name, parameter in items}
    tangent_dict = _unflatten(vector, items)
    tangents: list[tuple[torch.Tensor, torch.Tensor]] = []
    total = len(positions)
    real_sum = torch.zeros((), dtype=vector.dtype, device=vector.device)
    imag_sum = torch.zeros_like(real_sum)

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk_positions = positions[start:stop]
        chunk_layers = layers[start:stop]

        def scores(parameters):
            values = functional_call(
                wavefunction, parameters, (chunk_positions, chunk_layers), strict=False
            )
            return torch.cat([values.real, values.imag])

        _, tangent = jvp(scores, (parameter_dict,), (tangent_dict,))
        count = stop - start
        real_tangent = tangent[:count].detach()
        imag_tangent = tangent[count:].detach()
        tangents.append((real_tangent, imag_tangent))
        real_sum = real_sum + real_tangent.sum()
        imag_sum = imag_sum + imag_tangent.sum()

    real_mean = real_sum / total
    imag_mean = imag_sum / total
    output = torch.zeros_like(vector)
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        stop = min(start + chunk_size, total)
        chunk_positions = positions[start:stop]
        chunk_layers = layers[start:stop]
        real_tangent, imag_tangent = tangents[chunk_index]

        def scores(parameters):
            values = functional_call(
                wavefunction, parameters, (chunk_positions, chunk_layers), strict=False
            )
            return torch.cat([values.real, values.imag])

        _, pullback = vjp(scores, parameter_dict)
        cotangent = torch.cat([
            (real_tangent - real_mean) / total,
            (imag_tangent - imag_mean) / total,
        ])
        gradient_dict, = pullback(cotangent)
        output = output + torch.cat([
            gradient_dict[name].reshape(-1) for name, _ in items
        ]).detach()
    return output + damping * vector


def conjugate_gradient(
    matvec,
    right_hand_side: torch.Tensor,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, int, float]:
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    initial_norm = torch.sqrt(residual_squared).clamp_min(1.0e-30)
    iterations = 0
    for iteration in range(max_iterations):
        product = matvec(direction)
        denominator = torch.dot(direction, product)
        if not torch.isfinite(denominator) or denominator <= 0:
            break
        alpha = residual_squared / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_squared = torch.dot(residual, residual)
        iterations = iteration + 1
        if torch.sqrt(next_squared) <= tolerance * initial_norm:
            residual_squared = next_squared
            break
        beta = next_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = next_squared
    return solution, iterations, float(torch.sqrt(residual_squared))


def natural_gradient_step(
    wavefunction: torch.nn.Module,
    positions: torch.Tensor,
    layers: torch.Tensor,
    local_energy: torch.Tensor,
    learning_rate: float,
    damping: float,
    chunk_size: int,
    cg_iterations: int,
    cg_tolerance: float,
    maximum_direction_norm: float | None = None,
) -> NaturalGradientDiagnostics:
    """Take one matrix-free Sorella stochastic-reconfiguration step."""
    items = _parameter_items(wavefunction)
    force = energy_force(
        wavefunction, positions, layers, local_energy, chunk_size
    )

    def matvec(vector):
        return fisher_vector_product(
            wavefunction, positions, layers, vector, damping, chunk_size
        )

    direction, iterations, residual_norm = conjugate_gradient(
        matvec, force, cg_iterations, cg_tolerance
    )
    direction_norm = torch.linalg.norm(direction)
    clipped = False
    if maximum_direction_norm is not None and direction_norm > maximum_direction_norm:
        direction = direction * (maximum_direction_norm / direction_norm)
        direction_norm = torch.linalg.norm(direction)
        clipped = True
    with torch.no_grad():
        offset = 0
        for _, parameter in items:
            count = parameter.numel()
            parameter.add_(
                direction[offset:offset + count].reshape_as(parameter),
                alpha=-learning_rate,
            )
            offset += count
    return NaturalGradientDiagnostics(
        force_norm=float(torch.linalg.norm(force)),
        direction_norm=float(direction_norm),
        residual_norm=residual_norm,
        cg_iterations=iterations,
        direction_was_clipped=clipped,
    )
