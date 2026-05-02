"""
Physics-informed losses for heterogeneous primal velocity fields.

DEC-accurate sparse exterior derivatives can replace ``pseudo_divergence_loss``
once operators are exported onto PyG; today we penalise graph-gradient energy
plus a lightweight balanced-flow proxy on primal **vertex** subgraphs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def data_loss_mse(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target)


def filter_vertex_only_edges(edge_index: Tensor, n_vertices: int) -> Tensor:
    """Keep ``p2p`` edges whose endpoints lie in the primal vertex block."""
    mask = (edge_index[0] < n_vertices) & (edge_index[1] < n_vertices)
    return edge_index[:, mask]


def pseudo_divergence_loss(
    pred_vertex_features: Tensor,
    edge_index_p2p: Tensor,
    *,
    n_vertices: int,
    velocity_channels: tuple[int, int] = (0, 1),
) -> Tensor:
    """
    Cheap surrogate for ``∇·u ≈ 0`` without cotangent DEC weights.

    Combines (i) mean squared graph gradient magnitude on velocity channels and
    (ii) squared mean aggregated increments per vertex (discrete balance proxy).
    Returns ``0`` if no vertex-only edges survive filtering.
    """
    ei = filter_vertex_only_edges(edge_index_p2p, n_vertices)
    if ei.numel() == 0:
        return pred_vertex_features.new_zeros(())

    uv = pred_vertex_features[:n_vertices, [velocity_channels[0], velocity_channels[1]]]
    src, dst = ei
    du = uv[src] - uv[dst]
    grad_energy = du.pow(2).sum(dim=-1).mean()

    incr = du[:, 0] + du[:, 1]
    deg = torch.zeros(n_vertices, device=pred_vertex_features.device, dtype=pred_vertex_features.dtype)
    deg.index_add_(0, dst, torch.ones_like(incr))
    scatter_sum = torch.zeros(n_vertices, device=pred_vertex_features.device, dtype=pred_vertex_features.dtype)
    scatter_sum.index_add_(0, dst, incr)
    balance = scatter_sum / deg.clamp(min=1.0)
    balance_penalty = balance.pow(2).mean()

    return grad_energy + balance_penalty


def physics_informed_total_loss(
    pred_primal: Tensor,
    target_primal: Tensor,
    edge_index_p2p: Tensor,
    *,
    n_primal_vertices: int,
    lambda_physics: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    ``total = MSE + λ * Physics``.

    Physics term considers **only** primal vertices (fluid DOFs); data term spans
    all primal nodes (vertices + edge midpoints) so midpoints stay anchored.
    """
    data = data_loss_mse(pred_primal, target_primal)
    phys = pseudo_divergence_loss(
        pred_primal[:n_primal_vertices],
        edge_index_p2p,
        n_vertices=n_primal_vertices,
    )
    total = data + float(lambda_physics) * phys
    return total, data.detach(), phys.detach()
