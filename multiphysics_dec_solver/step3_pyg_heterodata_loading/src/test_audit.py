#!/usr/bin/env python3
"""
Build `HeteroData` from Step 2 V2 JSON and run pre-training audits.

Usage (from `step3_pyg_heterodata_loading/`):

    uv run --with torch --with torch-geometric --with numpy python src/test_audit.py

Or with pip-installed deps:

    python src/test_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hetero_dataset import HeteroV2Meta, load_v2_hetero_json, save_hetero_pt  # noqa: E402

DEFAULT_JSON = (
    ROOT.parent
    / "step2_heterogeneous_contract"
    / "data"
    / "v2_contract"
    / "hetero_cylinder_wake_t0.35.json"
)
OUT_PT = ROOT / "data" / "processed" / "hetero_cylinder_wake_t0.35.pt"


def _assert_edge_index(name: str, ei: torch.Tensor) -> None:
    assert ei.dtype == torch.long, f"{name}: edge_index must be torch.long"
    assert ei.dim() == 2 and ei.size(0) == 2, f"{name}: edge_index must be [2, E]"


def _assert_no_nan_inf(name: str, t: torch.Tensor) -> None:
    assert not torch.isnan(t).any(), f"{name}: NaN detected"
    assert not torch.isinf(t).any(), f"{name}: Inf detected"


def audit_hetero(data: HeteroData, meta: HeteroV2Meta) -> None:
    print("\n=== (1) HeteroData structure (PyG native print) ===\n")
    print(data)

    print("\n=== (2) NaN / Inf audit ===")
    for nt in ("primal", "dual"):
        pos = data[nt].pos
        x = data[nt].x
        _assert_no_nan_inf(f"{nt}.pos", pos)
        _assert_no_nan_inf(f"{nt}.x", x)
        assert pos.dtype == torch.float32
        assert x.dtype == torch.float32
        print(f"  OK: {nt}.pos / {nt}.x — no NaN or Inf")

    print("\n=== (3) Scale audit (mean / std) — primal vertices (physics features) ===")
    pv_x = data["primal"].x[: meta.n_primal_vertices]
    feat_names = ["velocity_u", "velocity_v", "pressure"]
    for i, fn in enumerate(feat_names):
        col = pv_x[:, i]
        print(f"  {fn}: mean={col.mean().item():.6e}, std={col.std(unbiased=False).item():.6e}")

    print("\n--- dual vertices (geom + placeholder pressure channel) ---")
    dv_x = data["dual"].x[: meta.n_dual_vertices]
    for i, fn in enumerate(["coord_x", "coord_y", "pressure_pad0"]):
        col = dv_x[:, i]
        print(f"  {fn}: mean={col.mean().item():.6e}, std={col.std(unbiased=False).item():.6e}")

    print("\n=== (4) dtype / shape asserts ===")
    for nt in ("primal", "dual"):
        assert data[nt].pos.dtype == torch.float32
        assert data[nt].x.dtype == torch.float32
        print(f"  OK: {nt} features are float32")

    _assert_edge_index(
        "p2p", data["primal", "p2p", "primal"].edge_index
    )
    _assert_edge_index(
        "d2d", data["dual", "d2d", "dual"].edge_index
    )
    _assert_edge_index(
        "p2d", data["primal", "p2d", "dual"].edge_index
    )
    print("  OK: all edge_index tensors are long and shaped [2, E]")

    # Index bounds (leakage-style sanity: no OOB references)
    np_total = data["primal"].num_nodes
    nd_total = data["dual"].num_nodes
    p2p = data["primal", "p2p", "primal"].edge_index
    d2d = data["dual", "d2d", "dual"].edge_index
    p2d = data["primal", "p2d", "dual"].edge_index
    if p2p.numel() > 0:
        assert int(p2p.max()) < meta.n_primal_vertices and int(p2p.min()) >= 0
    if d2d.numel() > 0:
        assert int(d2d.max()) < meta.n_dual_vertices and int(d2d.min()) >= 0
    if p2d.numel() > 0:
        assert int(p2d[0].max()) < np_total and int(p2d[0].min()) >= meta.n_primal_vertices
        assert int(p2d[1].max()) < nd_total and int(p2d[1].min()) >= meta.n_dual_vertices
    print("  OK: edge indices respect vertex vs edge-node partition")

    print("\n=== Audit completed successfully. ===\n")


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing Step 2 JSON: {json_path}")

    data, meta = load_v2_hetero_json(json_path)
    audit_hetero(data, meta)
    save_hetero_pt(data, meta, OUT_PT)
    print(f"Saved: {OUT_PT}")


if __name__ == "__main__":
    main()
