"""
Physics-informed 異種混合 GNN（Encoder / Processor / Decoder 分離）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

from dataset import DUAL, PRIMAL, E_D2P, E_DUAL, E_P2D, E_PRIMAL


def _reverse_edge_index(edge_index: Tensor) -> Tensor:
    return torch.stack([edge_index[1], edge_index[0]], dim=0)


class PhysicsHeteroGNN(nn.Module):
    """
    - Encoder: ノードタイプごとに Linear で隠れ次元へ。
    - Processor: HeteroConv + SAGEConv を複数層。p2d の逆辺を毎フォワードで生成し dual→primal も通す。
    - Decoder: 隠れ状態から primal / dual の次ステップ特徴（同一次元）を線形予測。
    """

    def __init__(
        self,
        in_dim_primal: int,
        in_dim_dual: int,
        out_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        *,
        sage_aggr: str = "mean",
        hetero_aggr: str = "sum",
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.num_layers = max(1, num_layers)

        self.encoder_primal = nn.Linear(in_dim_primal, hidden)
        self.encoder_dual = nn.Linear(in_dim_dual, hidden)

        convs: List[HeteroConv] = []
        for _ in range(self.num_layers):
            convs.append(
                HeteroConv(
                    {
                        E_PRIMAL: SAGEConv((-1, -1), hidden, aggr=sage_aggr),
                        E_DUAL: SAGEConv((-1, -1), hidden, aggr=sage_aggr),
                        E_P2D: SAGEConv((-1, -1), hidden, aggr=sage_aggr),
                        E_D2P: SAGEConv((-1, -1), hidden, aggr=sage_aggr),
                    },
                    aggr=hetero_aggr,
                )
            )
        self.convs = nn.ModuleList(convs)
        self.act = nn.ReLU(inplace=True)

        self.decoder_primal = nn.Linear(hidden, out_dim)
        self.decoder_dual = nn.Linear(hidden, out_dim)

    def _build_edge_index_dict(self, data_edge_index_dict: Dict[Tuple[str, str, str], Tensor]) -> Dict[Tuple[str, str, str], Tensor]:
        out = dict(data_edge_index_dict)
        if E_P2D in out:
            out[E_D2P] = _reverse_edge_index(out[E_P2D])
        return out

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        h: Dict[str, Tensor] = {
            PRIMAL: self.act(self.encoder_primal(x_dict[PRIMAL])),
            DUAL: self.act(self.encoder_dual(x_dict[DUAL])),
        }
        edges = self._build_edge_index_dict(edge_index_dict)
        for conv in self.convs:
            h = conv(h, edges)
            h = {k: self.act(v) for k, v in h.items()}
        return {
            PRIMAL: self.decoder_primal(h[PRIMAL]),
            DUAL: self.decoder_dual(h[DUAL]),
        }
