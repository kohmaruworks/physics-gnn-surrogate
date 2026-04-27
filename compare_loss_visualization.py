"""
基礎編 第3回（train_spring_mass_gcn.py）と同じ学習条件のもと、
2 層 GCN（グラフを利用）と平坦化 MLP（構造を無視）の学習損失を比較する。

- データ: `physics-gnn-surrogate-basic/spring_mass_chain_5.json`（同一リポ内に無い場合は
  親ディレクトリの sibling リポジトリを探索）
- 教師 y: Julia 側で ODE 積分した終端状態（第3回と同じ JSON）
- モデル: GCN — in=2, hidden=16, out=2（TwoLayerGCN 相当）
- MLP — 2 隠れ層、隠れ幅 16、入出力 N*2
- 最適化: Adam, lr=0.02, 100 エポック
- 指標: 毎エポックの**訓練** MSE（1 グラフ・第3回と同じ手順の比較）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from import_catlab_json_to_pyg import catlab_json_to_data


def undirected_spring_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """有向辺集合から、バネ 1 本につき 1 回だけ数える無向ペア (i, j), i < j。"""
    pairs: set[tuple[int, int]] = set()
    r0 = edge_index[0].tolist()
    r1 = edge_index[1].tolist()
    for s, t in zip(r0, r1):
        if s == t:
            continue
        a, b = (s, t) if s < t else (t, s)
        pairs.add((a, b))
    return sorted(pairs)


def spring_mass_next_state(
    x: torch.Tensor,
    spring_pairs: list[tuple[int, int]],
    *,
    k: float = 1.0,
    m: float = 1.0,
    dt: float = 0.05,
) -> torch.Tensor:
    """
    各ノード特徴: [位置 u, 速度 v]（1 次元を想定）。
    各無向バネ (i, j) について自然長 0 のフックの法則:
      ノード i への力 +k(u_j - u_i)、ノード j への力 -k(u_j - u_i)。
    合力から a = F/m、次ステップ: v' = v + a*dt, u' = u + v*dt。
    """
    device, dtype = x.device, x.dtype
    pos = x[:, 0]
    vel = x[:, 1]
    n = x.size(0)
    force = torch.zeros(n, device=device, dtype=dtype)
    for i, j in spring_pairs:
        du = pos[j] - pos[i]
        fi = k * du
        force[i] = force[i] + fi
        force[j] = force[j] - fi

    acc = force / m
    v_new = vel + acc * dt
    u_new = pos + vel * dt
    return torch.stack([u_new, v_new], dim=1)


class TwoLayerGCN(nn.Module):
    """第3回 `train_spring_mass_gcn.TwoLayerGCN` と同形: 2→16→2。"""

    def __init__(
        self, in_channels: int = 2, hidden_channels: int = 16, out_channels: int = 2
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class NaiveMLP(nn.Module):
    """全ノード特徴を平坦化。2 隠れ層、隠れ幅は GNN の 16 に合わせる。"""

    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        flat = num_nodes * feat_dim
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            flat = x.view(1, -1)
            out = self.net(flat)
            return out.view(self.num_nodes, self.feat_dim)
        num_graphs = int(batch.max().item()) + 1
        rows = []
        for g in range(num_graphs):
            mask = batch == g
            x_g = x[mask].view(1, -1)
            rows.append(self.net(x_g))
        flat_out = torch.cat(rows, dim=0)
        return flat_out.view(num_graphs * self.num_nodes, self.feat_dim)


def resolve_spring_chain_json(repo: Path) -> Path:
    """第3回と同じ `spring_mass_chain_5.json` を解決。"""
    candidates = [
        repo.parent / "physics-gnn-surrogate-basic" / "spring_mass_chain_5.json",
        repo / "spring_mass_chain_5.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "第3回と同じ教師付き JSON が見つかりません: spring_mass_chain_5.json\n"
        "次のいずれかに配置してください:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        "基礎編リポで `spring_mass_chain_export.jl` から生成できます。"
    )


def main() -> None:
    repo = Path(__file__).resolve().parent
    json_path = resolve_spring_chain_json(repo)
    data = catlab_json_to_data(json_path)
    if data.x is None or data.y is None:
        raise ValueError("JSON に x, y（教師）が必要です。第2回のエクスポート手順を参照。")

    num_nodes = int(data.num_nodes)
    feat_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    x = data.x
    y = data.y
    edge_index = data.edge_index

    hidden = 16
    lr = 0.02
    n_epochs = 100

    torch.manual_seed(7)
    gnn = TwoLayerGCN(
        in_channels=feat_dim, hidden_channels=hidden, out_channels=feat_dim
    ).to(device)
    mlp = NaiveMLP(num_nodes, feat_dim, hidden).to(device)

    opt_gnn = torch.optim.Adam(gnn.parameters(), lr=lr)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_gnn_hist: list[float] = []
    loss_mlp_hist: list[float] = []

    print(
        f"data: {json_path.name}  (same as Phase1 §3)\n"
        f"device={device}  num_nodes={num_nodes}  "
        f"TwoLayerGCN 2x{hidden}x2  MLP 2h/{hidden}  "
        f"Adam lr={lr}  epochs={n_epochs}\n"
    )

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        mlp.train()
        opt_gnn.zero_grad(set_to_none=True)
        opt_mlp.zero_grad(set_to_none=True)
        out_g = gnn(x, edge_index)
        out_m = mlp(x, None)
        loss_g = criterion(out_g, y)
        loss_m = criterion(out_m, y)
        loss_g.backward()
        loss_m.backward()
        opt_gnn.step()
        opt_mlp.step()
        loss_gnn_hist.append(float(loss_g.item()))
        loss_mlp_hist.append(float(loss_m.item()))
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(
                f"epoch {epoch:3d}  GNN train MSE={loss_gnn_hist[-1]:.6f}  "
                f"MLP train MSE={loss_mlp_hist[-1]:.6f}"
            )

    print(
        f"\n--- Final training MSE (epoch {n_epochs}) ---\n"
        f"  TwoLayerGCN: {loss_gnn_hist[-1]:.6f}\n"
        f"  Naive MLP:   {loss_mlp_hist[-1]:.6f}"
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    xs = range(1, n_epochs + 1)
    ax.plot(
        xs,
        loss_gnn_hist,
        color="#1565c0",
        linewidth=1.8,
        linestyle="-",
        label="TwoLayerGCN (train)",
    )
    ax.plot(
        xs,
        loss_mlp_hist,
        color="#c62828",
        linewidth=1.8,
        linestyle="--",
        label="Naive MLP (train)",
    )
    ax.set_title(
        "GNN vs MLP — training MSE (Phase 1 §3 same: spring_mass_chain_5.json, lr=0.02, 100 ep)",
        fontsize=11,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training MSE Loss", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.tight_layout()

    out_png = repo / "loss_comparison_test.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    zenn_images = repo.parent / "zenn-articles" / "images"
    if zenn_images.is_dir():
        zenn_out = zenn_images / "phase1-06-loss-comparison-test.png"
        fig.savefig(zenn_out, dpi=150, bbox_inches="tight")
        print(f"Saved figure (Zenn): {zenn_out}")

    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
