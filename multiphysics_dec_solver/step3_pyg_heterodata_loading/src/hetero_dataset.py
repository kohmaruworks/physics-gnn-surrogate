"""
Load Step 2 V2 heterogeneous JSON into PyTorch Geometric `HeteroData`.

JSON `primal_to_dual` relates **primal-edge IDs** to **dual-edge IDs**. PyG keys
``('primal', 'p2d', 'dual')`` require indices into `primal` and `dual` node stores,
so we **append** synthetic nodes at edge midpoints:
  - `primal`:  [ primal vertices | primal-edge midpoint nodes ]
  - `dual`:    [ dual vertices   | dual-edge midpoint nodes ]

Incidence edges `p2p` / `d2d` reference only the leading vertex blocks; `p2d`
offsets primal-edge and dual-edge IDs into those trailing blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class HeteroV2Meta:
    """Vertex counts for interpreting stacked node tensors."""

    n_primal_vertices: int
    n_dual_vertices: int
    n_primal_edges: int
    n_dual_edges: int


def _edge_midpoints_from_bidirected_coo(
    coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
    """Bidirected Julia COO: directed slot ``2*eid`` is (v0 -> v1)."""
    assert src.dim() == 1 and src.numel() % 2 == 0
    ne = src.numel() // 2
    mids = torch.empty((ne, coords.size(1)), dtype=torch.float32)
    for eid in range(ne):
        i = 2 * eid
        v0 = int(src[i].item())
        v1 = int(dst[i].item())
        mids[eid] = 0.5 * (coords[v0] + coords[v1])
    return mids


def load_v2_hetero_json(path: str | Path) -> tuple[HeteroData, HeteroV2Meta]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        j: dict[str, Any] = json.load(f)

    n_pv = int(j["nodes"]["primal"]["num_nodes"])
    n_dv = int(j["nodes"]["dual"]["num_nodes"])
    primal_coord = torch.tensor(j["nodes"]["primal"]["coordinates"], dtype=torch.float32)
    dual_coord = torch.tensor(j["nodes"]["dual"]["coordinates"], dtype=torch.float32)

    vu = torch.tensor(j["features"]["velocity_u"], dtype=torch.float32)
    vv = torch.tensor(j["features"]["velocity_v"], dtype=torch.float32)
    pr = torch.tensor(j["features"]["pressure"], dtype=torch.float32)
    if vu.shape[0] != n_pv or vv.shape[0] != n_pv or pr.shape[0] != n_pv:
        raise ValueError("Feature lengths must match primal vertex count")

    dec = j["dec_counts"]
    n_pe = int(dec["num_primal_edges"])
    n_de = int(dec["num_dual_edges"])

    pp = torch.tensor(j["edges"]["primal_to_primal"]["edge_index"], dtype=torch.long)
    dd = torch.tensor(j["edges"]["dual_to_dual"]["edge_index"], dtype=torch.long)
    pd_e = torch.tensor(j["edges"]["primal_to_dual"]["edge_index"], dtype=torch.long)

    if pp.numel() != 0 and pp.shape[0] != 2:
        raise ValueError("primal_to_primal.edge_index must be [2, E]")
    if dd.numel() != 0 and dd.shape[0] != 2:
        raise ValueError("dual_to_dual.edge_index must be [2, E]")
    if pd_e.numel() != 0 and pd_e.shape[0] != 2:
        raise ValueError("primal_to_dual.edge_index must be [2, E]")

    pe_mid = _edge_midpoints_from_bidirected_coo(primal_coord, pp[0], pp[1])
    de_mid = _edge_midpoints_from_bidirected_coo(dual_coord, dd[0], dd[1])
    if pe_mid.shape[0] != n_pe or de_mid.shape[0] != n_de:
        raise ValueError("dec_counts edge totals mismatch midpoint reconstruction")

    primal_pos = torch.cat([primal_coord, pe_mid], dim=0)
    dual_pos = torch.cat([dual_coord, de_mid], dim=0)

    primal_vertex_x = torch.stack([vu, vv, pr], dim=1)
    primal_edge_x = torch.zeros((n_pe, 3), dtype=torch.float32)
    primal_x = torch.cat([primal_vertex_x, primal_edge_x], dim=0)

    dual_vertex_x = torch.cat(
        [dual_coord.clone(), torch.zeros((n_dv, 1), dtype=torch.float32)], dim=1
    )
    dual_edge_x = torch.cat(
        [de_mid.clone(), torch.zeros((n_de, 1), dtype=torch.float32)], dim=1
    )
    dual_x = torch.cat([dual_vertex_x, dual_edge_x], dim=0)

    meta = HeteroV2Meta(
        n_primal_vertices=n_pv,
        n_dual_vertices=n_dv,
        n_primal_edges=n_pe,
        n_dual_edges=n_de,
    )

    data = HeteroData()
    data["primal"].pos = primal_pos
    data["primal"].x = primal_x
    data["dual"].pos = dual_pos
    data["dual"].x = dual_x

    data["primal", "p2p", "primal"].edge_index = pp
    data["dual", "d2d", "dual"].edge_index = dd

    if pd_e.numel() > 0:
        ps = pd_e[0] + n_pv
        ds = pd_e[1] + n_dv
        data["primal", "p2d", "dual"].edge_index = torch.stack([ps, ds], dim=0)
    else:
        z = torch.empty((2, 0), dtype=torch.long)
        data["primal", "p2d", "dual"].edge_index = z

    data.step2_json_path = str(path.resolve())

    return data, meta


def save_hetero_pt(
    data: HeteroData, meta: HeteroV2Meta, out_path: str | Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": data,
        "meta": meta,
    }
    torch.save(payload, out_path)
