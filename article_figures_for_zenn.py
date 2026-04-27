"""
Zenn 記事用: graph_from_catlab.json の PyG 復元結果を可視化して PNG を保存する。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from import_catlab_json_to_pyg import catlab_json_to_data


def _draw_chain_digraph(ax, num_nodes: int, edge_list: list[tuple[int, int]]) -> None:
    ax.set_xlim(-0.2, num_nodes - 0.2 + 0.6)
    ax.set_ylim(-0.35, 0.55)
    pos_x = {i: float(i) for i in range(num_nodes)}
    pos_y = {i: 0.12 for i in range(num_nodes)}
    r = 0.18
    for i in range(num_nodes):
        circ = plt.Circle((pos_x[i], pos_y[i]), r, color="#1a535c", ec="#00ffcc", lw=2)
        ax.add_patch(circ)
        ax.text(pos_x[i], pos_y[i], str(i), ha="center", va="center", color="white", fontsize=12)
    for s, t in edge_list:
        xs, ys, xt, yt = pos_x[s], pos_y[s], pos_x[t], pos_y[t]
        dx, dy = xt - xs, yt - ys
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-6:
            continue
        ux, uy = dx / dist, dy / dist
        start = (xs + ux * r, ys + uy * r)
        end = (xt - ux * r, yt - uy * r)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color="#7ae582", lw=1.8, shrinkA=0, shrinkB=0),
        )
    ax.axis("off")


def main() -> None:
    repo = Path(__file__).resolve().parent
    zenn_images = repo.parent / "zenn-articles" / "images"
    zenn_images.mkdir(parents=True, exist_ok=True)

    json_path = repo / "graph_from_catlab.json"
    data = catlab_json_to_data(json_path)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)

    n = int(data.num_nodes)
    ei = data.edge_index.cpu().tolist()
    pairs = list(zip(ei[0], ei[1]))
    _draw_chain_digraph(axes[0], n, pairs)
    axes[0].set_title("Restored topology (data.edge_index, 0-based)")

    x = data.x
    lines = [
        f"num_nodes = {data.num_nodes}",
        f"edge_index.shape = {tuple(data.edge_index.shape)}  dtype={data.edge_index.dtype}",
        f"x.shape = {tuple(x.shape)}  dtype={x.dtype}",
        "",
        "x (rows = nodes):",
    ]
    if x is not None:
        for i in range(x.size(0)):
            u, v = float(x[i, 0]), float(x[i, 1])
            lines.append(f"  node {i}:  u={u:.3f},  v={v:.3f}")
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        color="#e0e0e0",
    )
    axes[1].set_title("Tensor contract after catlab_json_to_data()")

    cyan = mpatches.Patch(color="#00ffcc", label="No Python-side index −1")
    axes[1].legend(handles=[cyan], loc="lower right", framealpha=0.3)

    fig.suptitle(
        "Phase 2 (ACT repo): JSON → torch_geometric.data.Data",
        fontsize=13,
        color="white",
    )
    fig.tight_layout()
    out_a = zenn_images / "phase1-04-json-to-pyg-restore.png"
    fig.savefig(out_a, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_a}")

    from compare_loss_visualization import (  # noqa: E402
        spring_mass_next_state,
        undirected_spring_pairs,
    )

    device = torch.device("cpu")
    spring_pairs = undirected_spring_pairs(data.edge_index)
    torch.manual_seed(0)
    x0 = torch.randn(n, 2, device=device) * 0.5
    y0 = spring_mass_next_state(x0, spring_pairs, k=1.0, m=1.0, dt=0.05)

    fig2, ax2 = plt.subplots(figsize=(5.5, 5), dpi=120)
    ax2.scatter(
        y0[:, 0].numpy(),
        y0[:, 1].numpy(),
        c="#00ffcc",
        s=80,
        alpha=0.85,
        label="Teacher y (one Euler step)",
    )
    ax2.set_xlabel(r"target $u'$ (position)")
    ax2.set_ylabel(r"target $v'$ (velocity)")
    ax2.set_title("Per-node targets on fixed graph (demo physics)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    out_b = zenn_images / "phase1-06-teacher-scatter-one-step.png"
    fig2.savefig(out_b, dpi=200, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Saved {out_b}")

    # --- Scale demo: MLP 入力次元の固定 vs GNN
    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=120)
    ax3.axis("off")
    ax3.set_title("Scale stress: flattened MLP vs message passing (concept)", fontsize=12)
    box_kws = dict(boxstyle="round,pad=0.35", facecolor="#2b2b2b", edgecolor="#00ffcc")
    ax3.text(
        0.5,
        0.72,
        r"MLP: $\mathbf{x}_{\mathrm{flat}}\in\mathbb{R}^{N_{\mathrm{design}}\,d}$\n"
        r"weights $W^{(1)}\in\mathbb{R}^{(N_{\mathrm{design}}d)\times h}$\n"
        r"$\Rightarrow$ at inference $N'\neq N_{\mathrm{design}}$ → matmul undefined",
        ha="center",
        va="center",
        fontsize=10,
        bbox=box_kws,
        color="#ff8888",
    )
    ax3.text(
        0.5,
        0.28,
        r"GCN: shared $W^{(\ell)}$ per layer; only $|\mathcal{E}|$ grows\n"
        r"$\Rightarrow$ same module, new edge_index for length-$N'$ chain",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#2b2b2b", edgecolor="#7ae582"),
        color="#a8e6cf",
    )
    fig3.tight_layout()
    out_c = zenn_images / "phase1-05-scale-mlp-vs-gcn-concept.png"
    fig3.savefig(out_c, dpi=200, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"Saved {out_c}")


if __name__ == "__main__":
    main()
