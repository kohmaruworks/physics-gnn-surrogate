#!/usr/bin/env python3
"""
Visualize Step 3 `HeteroData` checkpoint: local metapath subgraph + primal feature distributions.

Usage (from `step3_pyg_heterodata_loading/`):

    .venv/bin/python src/visualize_pyg.py
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hetero_dataset import HeteroV2Meta  # noqa: E402

DEFAULT_PT = ROOT / "data" / "processed" / "hetero_cylinder_wake_t0.35.pt"

ZENN = ROOT / "zenn_assets"
OUT_TOPO = ZENN / "pyg_subgraph_topology.png"
OUT_FEAT = ZENN / "pyg_feature_distributions.png"

COLOR_P2P = "#1f77b4"
COLOR_D2D = "#ff7f0e"
COLOR_P2D = "#2ca02c"

NodeKey = tuple[str, int]


def load_bundle(pt_path: Path) -> tuple[Any, HeteroV2Meta]:
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    return blob["data"], blob["meta"]


def _hetero_to_undirected_graph(data: Any) -> nx.Graph:
    """Undirected adjacency over keys ``('primal', i)`` and ``('dual', j)``."""
    G = nx.Graph()
    p2p = data["primal", "p2p", "primal"].edge_index
    d2d = data["dual", "d2d", "dual"].edge_index
    p2d = data["primal", "p2d", "dual"].edge_index

    def pair(u: NodeKey, v: NodeKey, et: str) -> None:
        if G.has_edge(u, v):
            G.edges[u, v]["types"].add(et)
        else:
            G.add_edge(u, v, types={et})

    if p2p.numel():
        for i in range(p2p.size(1)):
            u = ("primal", int(p2p[0, i].item()))
            v = ("primal", int(p2p[1, i].item()))
            pair(u, v, "p2p")
    if d2d.numel():
        for i in range(d2d.size(1)):
            u = ("dual", int(d2d[0, i].item()))
            v = ("dual", int(d2d[1, i].item()))
            pair(u, v, "d2d")
    if p2d.numel():
        for i in range(p2d.size(1)):
            u = ("primal", int(p2d[0, i].item()))
            v = ("dual", int(p2d[1, i].item()))
            pair(u, v, "p2d")
    return G


def _bfs_within_hops(G: nx.Graph, start: NodeKey, max_hops: int) -> set[NodeKey]:
    dist: dict[NodeKey, int] = {start: 0}
    q: deque[NodeKey] = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        if du == max_hops:
            continue
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return set(dist.keys())


def _pick_start_node(data: Any, meta: HeteroV2Meta, G: nx.Graph) -> NodeKey:
    start: NodeKey = ("primal", 0)
    if start in G:
        return start
    pv = data["primal"].pos[: meta.n_primal_vertices].numpy()
    centroid = pv.mean(axis=0)
    d2 = ((pv - centroid) ** 2).sum(axis=1)
    mid = int(d2.argmin())
    return ("primal", mid)


def plot_hetero_subgraph(
    data: Any,
    meta: HeteroV2Meta,
    out_path: Path,
    max_hops: int = 3,
    dpi: int = 300,
) -> None:
    G_full = _hetero_to_undirected_graph(data)
    start = _pick_start_node(data, meta, G_full)
    nodes = _bfs_within_hops(G_full, start, max_hops=max_hops)
    H = G_full.subgraph(nodes).copy()

    primal_pos = data["primal"].pos.numpy()
    dual_pos = data["dual"].pos.numpy()

    pos_xy: dict[NodeKey, tuple[float, float]] = {}
    for kind, idx in H.nodes():
        if kind == "primal":
            pos_xy[(kind, idx)] = (float(primal_pos[idx, 0]), float(primal_pos[idx, 1]))
        else:
            pos_xy[(kind, idx)] = (float(dual_pos[idx, 0]), float(dual_pos[idx, 1]))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)

    def edge_color(types: set[str]) -> str:
        if "p2d" in types:
            return COLOR_P2D
        if "p2p" in types:
            return COLOR_P2P
        if "d2d" in types:
            return COLOR_D2D
        return "#7f7f7f"

    by_c: dict[str, list[tuple[NodeKey, NodeKey]]] = defaultdict(list)
    for u, v, ed in H.edges(data=True):
        col = edge_color(ed.get("types", set()))
        by_c[col].append((u, v))

    edge_handles = []
    edge_labels = []
    for col, lab in [(COLOR_P2P, "p2p"), (COLOR_D2D, "d2d"), (COLOR_P2D, "p2d")]:
        pairs = by_c.get(col, [])
        if not pairs:
            continue
        nx.draw_networkx_edges(
            H,
            pos_xy,
            edgelist=pairs,
            edge_color=col,
            width=1.4,
            alpha=0.88,
            ax=ax,
        )
        edge_handles.append(mlines.Line2D([0], [0], color=col, lw=2.5))
        edge_labels.append(lab)

    def node_bucket(k: NodeKey) -> str:
        kind, idx = k
        if kind == "primal":
            return "pv" if idx < meta.n_primal_vertices else "pe"
        return "dv" if idx < meta.n_dual_vertices else "de"

    buckets = {"pv": [], "pe": [], "dv": [], "de": []}
    for n in H.nodes():
        buckets[node_bucket(n)].append(n)

    nx.draw_networkx_nodes(
        H,
        pos_xy,
        nodelist=buckets["pv"],
        node_color="#4169e1",
        node_shape="o",
        node_size=130,
        edgecolors="white",
        linewidths=0.6,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        H,
        pos_xy,
        nodelist=buckets["pe"],
        node_color="#87ceeb",
        node_shape="^",
        node_size=115,
        edgecolors="navy",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        H,
        pos_xy,
        nodelist=buckets["dv"],
        node_color="#dc143c",
        node_shape="s",
        node_size=100,
        edgecolors="white",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        H,
        pos_xy,
        nodelist=buckets["de"],
        node_color="#ffb347",
        node_shape="D",
        node_size=95,
        edgecolors="darkred",
        linewidths=0.45,
        ax=ax,
    )

    node_handles = [
        mlines.Line2D(
            [],
            [],
            color="#4169e1",
            marker="o",
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            label="Primal vertex",
        ),
        mlines.Line2D(
            [],
            [],
            color="#87ceeb",
            marker="^",
            linestyle="None",
            markersize=9,
            markeredgecolor="navy",
            label="Primal edge midpoint",
        ),
        mlines.Line2D(
            [],
            [],
            color="#dc143c",
            marker="s",
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            label="Dual vertex",
        ),
        mlines.Line2D(
            [],
            [],
            color="#ffb347",
            marker="D",
            linestyle="None",
            markersize=7,
            markeredgecolor="darkred",
            label="Dual edge midpoint",
        ),
    ]

    ax.set_title(
        f"PyG metapath subgraph (≤{max_hops} hops from center {start}; "
        f"|V|={H.number_of_nodes()}, |E|={H.number_of_edges()})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    leg_nodes = ax.legend(
        handles=node_handles,
        loc="upper left",
        fontsize=8,
        title="Nodes",
        framealpha=0.92,
    )
    ax.add_artist(leg_nodes)
    if edge_handles:
        ax.legend(
            handles=edge_handles,
            labels=edge_labels,
            loc="lower left",
            fontsize=8,
            title="Edges",
            framealpha=0.92,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(
    data: Any,
    meta: HeteroV2Meta,
    out_path: Path,
    dpi: int = 300,
) -> None:
    x = data["primal"].x[: meta.n_primal_vertices].numpy()
    names = ["velocity_u", "velocity_v", "pressure"]
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), dpi=dpi)
    for ax, j, name in zip(axes, range(min(x.shape[1], len(names))), names):
        sns.histplot(x[:, j], kde=True, ax=ax, color="#4c72b0", edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.suptitle("Primal vertex input features (before NN normalization)", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PT
    if not pt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {pt}")
    data, meta = load_bundle(pt)
    plot_hetero_subgraph(data, meta, OUT_TOPO)
    plot_feature_distributions(data, meta, OUT_FEAT)
    print(f"Wrote {OUT_TOPO}")
    print(f"Wrote {OUT_FEAT}")


if __name__ == "__main__":
    main()
