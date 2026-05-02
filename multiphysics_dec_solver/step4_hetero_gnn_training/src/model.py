"""
Heterogeneous GNN built from ``HeteroConv`` + ``GraphConv`` message passing.

Primal / Dual complexes use distinct convolution stacks per metapath;
reverse dual→primal edges are injected upstream so both propagation directions exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear
from torch_geometric.nn.conv import GraphConv


def augment_reverse_edges(data: HeteroData) -> HeteroData:
    """Clone ``data`` and attach reverse ``('dual','d2p','primal')`` edges from ``p2d``."""
    out = data.clone()
    p2d = out["primal", "p2d", "dual"].edge_index
    if p2d.numel() > 0:
        out["dual", "d2p", "primal"].edge_index = p2d.flip(0)
    else:
        out["dual", "d2p", "primal"].edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=p2d.device
        )
    return out


class PhysicsInformedHeteroGNN(nn.Module):
    """
    Stack ``HeteroConv(GraphConv, ...)`` layers then Linear heads per node store.

    Returns primal predictions shaped ``[num_primal_nodes, primal_out_dim]``.
    """

    def __init__(
        self,
        primal_in_dim: int,
        dual_in_dim: int,
        hidden_dim: int,
        primal_out_dim: int,
        num_layers: int,
        *,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.primal_in_dim = primal_in_dim
        self.dual_in_dim = dual_in_dim
        self.hidden_dim = hidden_dim
        self.primal_out_dim = primal_out_dim
        self.num_layers = max(1, num_layers)

        self.convs = nn.ModuleList()
        self.norm_primal = nn.ModuleList()
        self.norm_dual = nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                cc = {
                    ("primal", "p2p", "primal"): GraphConv(
                        (primal_in_dim, primal_in_dim), hidden_dim, aggr=aggr
                    ),
                    ("dual", "d2d", "dual"): GraphConv(
                        (dual_in_dim, dual_in_dim), hidden_dim, aggr=aggr
                    ),
                    ("primal", "p2d", "dual"): GraphConv(
                        (primal_in_dim, dual_in_dim), hidden_dim, aggr=aggr
                    ),
                    ("dual", "d2p", "primal"): GraphConv(
                        (dual_in_dim, primal_in_dim), hidden_dim, aggr=aggr
                    ),
                }
            else:
                cc = {
                    ("primal", "p2p", "primal"): GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr=aggr
                    ),
                    ("dual", "d2d", "dual"): GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr=aggr
                    ),
                    ("primal", "p2d", "dual"): GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr=aggr
                    ),
                    ("dual", "d2p", "primal"): GraphConv(
                        (hidden_dim, hidden_dim), hidden_dim, aggr=aggr
                    ),
                }
            self.convs.append(HeteroConv(cc, aggr="sum"))
            self.norm_primal.append(nn.LayerNorm(hidden_dim))
            self.norm_dual.append(nn.LayerNorm(hidden_dim))

        self.act = nn.ReLU(inplace=True)
        self.lin_primal = Linear(hidden_dim, primal_out_dim)

    def forward(self, data: HeteroData, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        xd: dict[str, torch.Tensor] = {k: v for k, v in x_dict.items()}
        for conv, nrm_p, nrm_d in zip(self.convs, self.norm_primal, self.norm_dual):
            xd = conv(xd, data.edge_index_dict)
            xd["primal"] = self.act(nrm_p(xd["primal"]))
            xd["dual"] = self.act(nrm_d(xd["dual"]))
        return self.lin_primal(xd["primal"])
