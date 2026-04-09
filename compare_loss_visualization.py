"""
Catlab 由来のグラフ上で、圏論構造を使う GNN と構造を無視する MLP の
汎化性能（テスト損失）を公平な条件下で比較する。

正解ラベルはバネ–質量系のフックの法則に基づく合力から、
オイラー積分で得た次ステップの状態 [位置, 速度] とする（辺の重複は 1 バネとして数える）。

200 組の (x, y) を生成し 150 Train / 50 Test に分割。
学習は Train のみ。グラフには毎エポックの Test MSE のみをプロットする。

※ 本ターゲットは状態 x に対して線形写像になる。平坦化 MLP は十分な幅があれば
   任意の線形写像に近づけやすく、重み共有する GCN は表現クラスが異なり学習曲線が違う。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
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


class CategoryInformedGNN(nn.Module):
    """隠れ次元 hidden。2×GCN + 線形（MLP 側の 2 隠れ層と同じ段数の目安）。"""

    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.lin(h)


class NaiveMLP(nn.Module):
    """全ノード特徴を平坦化。2 隠れ層 + 出力（GNN と同じ hidden 次元）。"""

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


def build_fixed_dataset(
    base: Data,
    spring_pairs: list[tuple[int, int]],
    *,
    n_samples: int,
    num_nodes: int,
    feat_dim: int,
    device: torch.device,
    pos_scale: float,
    vel_scale: float,
    k_spring: float,
    mass: float,
    dt_phys: float,
    generator: torch.Generator | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for _ in range(n_samples):
        x_s = torch.zeros(num_nodes, feat_dim, device=device)
        x_s[:, 0] = torch.randn(num_nodes, device=device, generator=generator) * pos_scale
        x_s[:, 1] = torch.randn(num_nodes, device=device, generator=generator) * vel_scale
        y_s = spring_mass_next_state(
            x_s, spring_pairs, k=k_spring, m=mass, dt=dt_phys
        )
        xs.append(x_s)
        ys.append(y_s)
    return xs, ys


def make_batch(
    base: Data,
    x_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
) -> Batch:
    data_list = []
    for xa, ya in zip(x_list, y_list):
        data_list.append(
            Data(
                x=xa,
                y=ya,
                edge_index=base.edge_index,
                num_nodes=base.num_nodes,
            )
        )
    return Batch.from_data_list(data_list)


@torch.no_grad()
def eval_mse(
    model: nn.Module, batch: Batch, criterion: nn.Module, *, use_gnn: bool
) -> float:
    if use_gnn:
        pred = model(batch.x, batch.edge_index)
    else:
        pred = model(batch.x, batch.batch)
    return float(criterion(pred, batch.y).item())


def main() -> None:
    repo = Path(__file__).resolve().parent
    json_path = repo / "graph_from_catlab.json"
    base = catlab_json_to_data(json_path)

    num_nodes = int(base.num_nodes)
    feat_dim = 2
    edge_index = base.edge_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    spring_pairs = undirected_spring_pairs(base.edge_index)

    torch.manual_seed(42)
    dataset_seed = torch.Generator(device=device)
    dataset_seed.manual_seed(42)

    # 隠れ次元を両者で同一（パラメータ数はアーキテクチャのため一致しない）
    hidden = 96

    gnn = CategoryInformedGNN(feat_dim, hidden, feat_dim).to(device)
    mlp = NaiveMLP(num_nodes, feat_dim, hidden).to(device)

    lr = 1e-3
    opt_gnn = torch.optim.Adam(gnn.parameters(), lr=lr)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = 800
    n_total = 200
    n_train = 150
    n_test = 50
    pos_scale = 1.0
    vel_scale = 0.5
    k_spring, mass, dt_phys = 1.0, 1.0, 0.05

    x_all, y_all = build_fixed_dataset(
        base,
        spring_pairs,
        n_samples=n_total,
        num_nodes=num_nodes,
        feat_dim=feat_dim,
        device=device,
        pos_scale=pos_scale,
        vel_scale=vel_scale,
        k_spring=k_spring,
        mass=mass,
        dt_phys=dt_phys,
        generator=dataset_seed,
    )

    split_gen = torch.Generator()
    split_gen.manual_seed(42)
    perm = torch.randperm(n_total, generator=split_gen).tolist()
    train_idx = perm[:n_train]
    test_idx = perm[n_train : n_train + n_test]

    x_train = [x_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    x_test = [x_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

    train_batch = make_batch(base, x_train, y_train).to(device)
    test_batch = make_batch(base, x_test, y_test).to(device)

    test_loss_gnn: list[float] = []
    test_loss_mlp: list[float] = []

    for ep in range(1, epochs + 1):
        gnn.train()
        mlp.train()

        opt_gnn.zero_grad()
        loss_g_train = criterion(
            gnn(train_batch.x, train_batch.edge_index), train_batch.y
        )
        loss_g_train.backward()
        opt_gnn.step()

        opt_mlp.zero_grad()
        loss_m_train = criterion(mlp(train_batch.x, train_batch.batch), train_batch.y)
        loss_m_train.backward()
        opt_mlp.step()

        gnn.eval()
        mlp.eval()
        test_loss_gnn.append(eval_mse(gnn, test_batch, criterion, use_gnn=True))
        test_loss_mlp.append(eval_mse(mlp, test_batch, criterion, use_gnn=False))

        if ep % 20 == 0:
            print(
                f"epoch {ep:4d}  GNN test={test_loss_gnn[-1]:.6f}  "
                f"MLP test={test_loss_mlp[-1]:.6f}"
            )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    epochs_axis = range(1, len(test_loss_gnn) + 1)
    ax.plot(
        epochs_axis,
        test_loss_gnn,
        color="#00ffcc",
        linewidth=2.5,
        linestyle="-",
        label="Category-informed GNN (test)",
    )
    ax.plot(
        epochs_axis,
        test_loss_mlp,
        color="#ff4444",
        linewidth=2.0,
        linestyle=":",
        label="Naive MLP (test)",
    )
    ax.set_title(
        "Physics Surrogate: Test Loss — Generalization Comparison", fontsize=14
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png = repo / "loss_comparison_test.png"
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
