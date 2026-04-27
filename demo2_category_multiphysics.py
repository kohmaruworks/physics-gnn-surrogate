"""
マルチフィジックス（バネ + ダンパー）における
同質 GNN と HeteroData による圏論的スキーマ分離 GNN の学習比較デモ。

正解はスプリング辺で F ∝ (u_j - u_i)、ダンパー辺で F ∝ (v_j - v_i) を足し合わせ、
a = F/m とオイラーで次状態を計算。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, HeteroConv


def chain_spring_damper_topology(num_nodes: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    偶数ノードの直列チェーン: (0-1),(2-3),… をバネ、(1-2),(3-4),… をダンパー。
    """
    if num_nodes < 4 or num_nodes % 2 != 0:
        raise ValueError("num_nodes must be even and >= 4")
    springs: list[tuple[int, int]] = []
    dampers: list[tuple[int, int]] = []
    for i in range(0, num_nodes - 1, 2):
        springs.append((i, i + 1))
    for i in range(1, num_nodes - 1, 2):
        dampers.append((i, i + 1))
    return springs, dampers


def pairs_to_bidirectional(ei: torch.Tensor) -> torch.Tensor:
    """[[i,j],...] -> PyG edge_index (2, 2E) with both directions."""
    if ei.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    row = ei[0].tolist()
    col = ei[1].tolist()
    src = row + col
    tgt = col + row
    return torch.tensor([src, tgt], dtype=torch.long)


def list_pairs_to_edge_index(
    pairs: list[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ei = torch.tensor(
        [[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long, device=device
    )
    return pairs_to_bidirectional(ei).to(device)


def multiphysics_next_state(
    x: torch.Tensor,
    spring_pairs: list[tuple[int, int]],
    damper_pairs: list[tuple[int, int]],
    *,
    k: float,
    c: float,
    m: float,
    dt: float,
) -> torch.Tensor:
    """ノード特徴 [位置 u, 速度 v]。バネは変位、ダンパーは速度差に比例する力を加算。"""
    device, dtype = x.device, x.dtype
    n = x.size(0)
    pos = x[:, 0]
    vel = x[:, 1]
    force = torch.zeros(n, device=device, dtype=dtype)

    for i, j in spring_pairs:
        du = pos[j] - pos[i]
        fi = k * du
        force[i] = force[i] + fi
        force[j] = force[j] - fi

    for i, j in damper_pairs:
        dv = vel[j] - vel[i]
        fi = c * dv
        force[i] = force[i] + fi
        force[j] = force[j] - fi

    acc = force / m
    v_new = vel + acc * dt
    u_new = pos + vel * dt
    return torch.stack([u_new, v_new], dim=1)


class HomogeneousGNN(nn.Module):
    """スプリング・ダンパー辺を同一 edge_index に混在させた同質 GCN。"""

    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.lin(h)


class CategoryHeteroGNN(nn.Module):
    """('node','spring','node') / ('node','damper','node') を HeteroConv で分離。"""

    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("node", "spring", "node"): GCNConv(-1, hidden),
                ("node", "damper", "node"): GCNConv(-1, hidden),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("node", "spring", "node"): GCNConv(-1, hidden),
                ("node", "damper", "node"): GCNConv(-1, hidden),
            },
            aggr="sum",
        )
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {"node": data["node"].x}
        edge_index_dict = {
            ("node", "spring", "node"): data["node", "spring", "node"].edge_index,
            ("node", "damper", "node"): data["node", "damper", "node"].edge_index,
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict["node"] = F.relu(x_dict["node"])
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict["node"] = F.relu(x_dict["node"])
        return self.lin(x_dict["node"])


def make_hetero_data(
    x: torch.Tensor,
    edge_index_spring: torch.Tensor,
    edge_index_damper: torch.Tensor,
) -> HeteroData:
    d = HeteroData()
    d["node"].x = x
    d["node", "spring", "node"].edge_index = edge_index_spring
    d["node", "damper", "node"].edge_index = edge_index_damper
    return d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    num_nodes = 12
    feat_dim = 2
    hidden = 64
    lr = 1e-3
    epochs = 200

    k_spring = 1.0
    c_damper = 0.8
    mass = 1.0
    dt_phys = 0.05
    pos_scale = 1.0
    vel_scale = 0.6

    spring_pairs, damper_pairs = chain_spring_damper_topology(num_nodes)
    ei_spring = list_pairs_to_edge_index(spring_pairs, device)
    ei_damper = list_pairs_to_edge_index(damper_pairs, device)
    edge_index_homo = torch.cat([ei_spring, ei_damper], dim=1)

    homo = HomogeneousGNN(feat_dim, hidden, feat_dim).to(device)
    hetero = CategoryHeteroGNN(hidden, feat_dim).to(device)

    opt_h = torch.optim.Adam(homo.parameters(), lr=lr)
    opt_ht = torch.optim.Adam(hetero.parameters(), lr=lr)
    crit = nn.MSELoss()

    loss_homo_hist: list[float] = []
    loss_hetero_hist: list[float] = []

    pool_size = 64
    pool_x: list[torch.Tensor] = []
    pool_y: list[torch.Tensor] = []
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    for _ in range(pool_size):
        xt = torch.zeros(num_nodes, feat_dim, device=device)
        xt[:, 0] = torch.randn(num_nodes, device=device, generator=g) * pos_scale
        xt[:, 1] = torch.randn(num_nodes, device=device, generator=g) * vel_scale
        yt = multiphysics_next_state(
            xt,
            spring_pairs,
            damper_pairs,
            k=k_spring,
            c=c_damper,
            m=mass,
            dt=dt_phys,
        )
        pool_x.append(xt)
        pool_y.append(yt)

    print(
        f"device={device}  nodes={num_nodes}  springs={len(spring_pairs)}  "
        f"dampers={len(damper_pairs)}  epochs={epochs}  lr={lr}  "
        f"fixed_pool={pool_size}"
    )

    for ep in range(1, epochs + 1):
        opt_h.zero_grad()
        opt_ht.zero_grad()
        loss_h_acc = torch.zeros((), device=device)
        loss_ht_acc = torch.zeros((), device=device)
        for x, y in zip(pool_x, pool_y):
            data_homo = Data(x=x, edge_index=edge_index_homo)
            pred_h = homo(data_homo.x, data_homo.edge_index)
            loss_h_acc = loss_h_acc + crit(pred_h, y)
            hdata = make_hetero_data(x, ei_spring, ei_damper)
            pred_ht = hetero(hdata)
            loss_ht_acc = loss_ht_acc + crit(pred_ht, y)
        loss_h = loss_h_acc / pool_size
        loss_ht = loss_ht_acc / pool_size
        loss_h.backward()
        loss_ht.backward()
        opt_h.step()
        opt_ht.step()

        loss_homo_hist.append(loss_h.item())
        loss_hetero_hist.append(loss_ht.item())

        if ep % 40 == 0:
            print(
                f"epoch {ep:3d}  Homo loss={loss_h.item():.6f}  "
                f"Hetero loss={loss_ht.item():.6f}"
            )

    # 最終エポックの損失を表示
    print(
        f"\n--- Final train MSE ---\n"
        f"  Category Hetero GNN : {loss_hetero_hist[-1]:.6f}\n"
        f"  Homogeneous GNN     : {loss_homo_hist[-1]:.6f}"
    )

    repo = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    xs = range(1, epochs + 1)
    ax.plot(
        xs,
        loss_hetero_hist,
        color="#1565c0",
        linewidth=1.8,
        linestyle="-",
        label="Category Hetero GNN (spring | damper separated)",
    )
    ax.plot(
        xs,
        loss_homo_hist,
        color="#c62828",
        linewidth=1.8,
        linestyle="--",
        label="Homogeneous GNN (mixed edges)",
    )
    ax.set_title(
        "Multiphysics Learning: Schema-Separated Hetero GNN vs Mixed-Edge GCN", fontsize=12
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train MSE Loss", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.tight_layout()

    out = repo / "hetero_loss_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    zenn_images = repo.parent / "zenn-articles" / "images"
    if zenn_images.is_dir():
        zenn_out = zenn_images / "phase1-06-hetero-vs-homo-mse.png"
        fig.savefig(zenn_out, dpi=150, bbox_inches="tight")
        print(f"Saved figure (Zenn): {zenn_out}")

    plt.close(fig)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
