"""
Catlab 由来の JSON（export_catlab_graph_json.jl が出力する形式）を
torch_geometric.data.Data に変換する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def catlab_json_to_data(path: str | Path, *, dtype_edge_index=torch.int64) -> Data:
    """
    JSON 形式:
      - format: "catlab_directed_graph_v1"
      - num_nodes: int（孤立頂点を含む頂点数）
      - edges: [[src, tgt], ...]（0-based）
      - x: 任意。[[...], ...] 形状 (num_nodes, feat_dim)
      - y: 任意。ノードターゲット（例: 物理シミュレーションの終端状態）
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if payload.get("format") != "catlab_directed_graph_v1":
        raise ValueError(
            f"unsupported format: {payload.get('format')!r}; "
            "expected 'catlab_directed_graph_v1'"
        )

    num_nodes = int(payload["num_nodes"])
    edges = payload.get("edges") or []

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=dtype_edge_index)
    else:
        edge_index = torch.tensor(edges, dtype=dtype_edge_index).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    if "x" in payload:
        data.x = torch.tensor(payload["x"], dtype=torch.float32)

    if "y" in payload:
        data.y = torch.tensor(payload["y"], dtype=torch.float32)

    return data


if __name__ == "__main__":
    import sys

    default_json = Path(__file__).resolve().parent / "graph_from_catlab.json"
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    d = catlab_json_to_data(p)
    print(d)
    print("edge_index:\n", d.edge_index)
    if d.x is not None:
        print("x:\n", d.x)
