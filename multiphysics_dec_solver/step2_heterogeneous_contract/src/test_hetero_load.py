#!/usr/bin/env python3
"""
Load Step 2 heterogeneous JSON Contract V2, run 0-based index sanity checks,
and plot primal/dual vertex topology with sampled primal↔dual (edge–edge) segments.

Run from repo root or from step2_heterogeneous_contract:
    pip install -r requirements_test.txt
    python src/test_hetero_load.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "data" / "v2_contract" / "hetero_cylinder_wake_t0.35.json"
OUT_FIG = ROOT / "zenn_assets" / "hetero_topology.png"


def load_contract(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def edge_midpoints_from_bidirected_coo(
    coords: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Bidirected COO from Julia: for edge id k, slot 2*k is (v0, v1)."""
    nslots = src.shape[0]
    assert nslots % 2 == 0
    ne = nslots // 2
    mids = np.zeros((ne, 2), dtype=np.float64)
    for eid in range(ne):
        i = 2 * eid
        v0, v1 = int(src[i]), int(dst[i])
        mids[eid] = 0.5 * (coords[v0] + coords[v1])
    return mids


def print_and_validate(data: dict) -> None:
    nodes = data["nodes"]
    edges = data["edges"]
    dec = data.get("dec_counts", {})

    n_primal = int(nodes["primal"]["num_nodes"])
    n_dual = int(nodes["dual"]["num_nodes"])
    primal_coords = np.asarray(nodes["primal"]["coordinates"], dtype=np.float64)
    dual_coords = np.asarray(nodes["dual"]["coordinates"], dtype=np.float64)

    print("=== Step 2 Heterogeneous JSON — sanity check ===")
    print(f"Primal vertices (num_nodes): {n_primal}")
    print(f"Dual vertices   (num_nodes): {n_dual}")
    assert primal_coords.shape == (n_primal, 2)
    assert dual_coords.shape == (n_dual, 2)

    num_pe = dec.get("num_primal_edges")
    num_de = dec.get("num_dual_edges")
    if num_pe is None or num_de is None:
        raise AssertionError("dec_counts.num_primal_edges / num_dual_edges required for validation")

    for name in ("primal_to_primal", "dual_to_dual", "primal_to_dual"):
        ei = edges[name]["edge_index"]
        src = np.asarray(ei[0], dtype=np.int64)
        dst = np.asarray(ei[1], dtype=np.int64)
        m = src.shape[0]
        print(f"\n--- {name} ---")
        print(f"  Edge count (directed COO rows): {m}")
        if m == 0:
            print("  (empty)")
            continue
        print(f"  Source index min/max: {src.min()} / {src.max()}")
        print(f"  Target index min/max: {dst.min()} / {dst.max()}")

        assert np.all(src >= 0) and np.all(dst >= 0), f"{name}: negative index"

        if name == "primal_to_primal":
            assert np.all(src < n_primal) and np.all(dst < n_primal), (
                f"{name}: index out of primal vertex range [0, {n_primal - 1}]"
            )
            assert m == 2 * num_pe, (
                f"{name}: expected 2*num_primal_edges directed rows, got {m} vs 2*{num_pe}"
            )
        elif name == "dual_to_dual":
            assert np.all(src < n_dual) and np.all(dst < n_dual), (
                f"{name}: index out of dual vertex range [0, {n_dual - 1}]"
            )
            assert m == 2 * num_de, (
                f"{name}: expected 2*num_dual_edges directed rows, got {m} vs 2*{num_de}"
            )
        else:
            # bipartite relation on edge IDs — not vertex IDs
            assert np.all(src < num_pe) and np.all(dst < num_de), (
                f"{name}: src must be < num_primal_edges ({num_pe}), "
                f"dst < num_dual_edges ({num_de})"
            )

    print("\n=== All index assertions passed (0-based, in-range). ===")


def plot_topology(data: dict, out_path: Path, max_pd_segments: int = 1000) -> None:
    nodes = data["nodes"]
    edges = data["edges"]
    dec = data["dec_counts"]

    primal_xy = np.asarray(nodes["primal"]["coordinates"], dtype=np.float64)
    dual_xy = np.asarray(nodes["dual"]["coordinates"], dtype=np.float64)

    pp_src = np.asarray(edges["primal_to_primal"]["edge_index"][0], dtype=np.int64)
    pp_dst = np.asarray(edges["primal_to_primal"]["edge_index"][1], dtype=np.int64)
    dd_src = np.asarray(edges["dual_to_dual"]["edge_index"][0], dtype=np.int64)
    dd_dst = np.asarray(edges["dual_to_dual"]["edge_index"][1], dtype=np.int64)
    pd_src = np.asarray(edges["primal_to_dual"]["edge_index"][0], dtype=np.int64)
    pd_dst = np.asarray(edges["primal_to_dual"]["edge_index"][1], dtype=np.int64)

    primal_edge_mid = edge_midpoints_from_bidirected_coo(primal_xy, pp_src, pp_dst)
    dual_edge_mid = edge_midpoints_from_bidirected_coo(dual_xy, dd_src, dd_dst)

    assert primal_edge_mid.shape[0] == dec["num_primal_edges"]
    assert dual_edge_mid.shape[0] == dec["num_dual_edges"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_aspect("equal", adjustable="box")

    n_draw = min(max_pd_segments, pd_src.shape[0])
    idx = np.arange(n_draw)
    segs = np.stack(
        [primal_edge_mid[pd_src[idx]], dual_edge_mid[pd_dst[idx]]], axis=1
    )
    lc = plt.matplotlib.collections.LineCollection(
        segs, colors="0.65", linewidths=0.6, alpha=0.3
    )
    ax.add_collection(lc)

    ax.scatter(
        primal_xy[:, 0],
        primal_xy[:, 1],
        s=8,
        c="tab:blue",
        marker="o",
        label="Primal vertices",
        zorder=3,
        edgecolors="none",
    )
    ax.scatter(
        dual_xy[:, 0],
        dual_xy[:, 1],
        s=22,
        c="tab:red",
        marker="x",
        label="Dual vertices",
        zorder=4,
        linewidths=0.8,
    )

    ax.set_title("Categorical Hetero-Graph: Primal & Dual Nodes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path} (primal_to_dual segments drawn: {n_draw})")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    if not path.is_file():
        sys.stderr.write(f"JSON not found: {path}\n")
        sys.exit(1)
    data = load_contract(path)
    print_and_validate(data)
    plot_topology(data, OUT_FIG)


if __name__ == "__main__":
    main()
