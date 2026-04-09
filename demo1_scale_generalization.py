"""
デモ: ノード数可変に対する Zero-Shot 汎化（GNN は実行可能、固定入力 MLP は崩壊）。

学習は行わず N=10 用に初期化したモデルに、N=50 のデータを流し込む。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def chain_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    """1 次元直列チェーンの有向辺 (i,i+1) のみ（compare_loss と同様の片方向バネ列）。"""
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src: list[int] = []
    tgt: list[int] = []
    for i in range(num_nodes - 1):
        src.append(i)
        tgt.append(i + 1)
    return torch.tensor([src, tgt], dtype=torch.long, device=device)


class CategoryInformedGNN(nn.Module):
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
    """ノード数 × 特徴次元が固定。異なる N には対応できない。"""

    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        flat = num_nodes * feat_dim
        self.net = nn.Sequential(
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(1, -1)
        return self.net(flat).view(self.num_nodes, self.feat_dim)


def random_node_features(num_nodes: int, feat_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num_nodes, feat_dim, device=device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    n_train = 10
    n_test = 50
    feat_dim = 2
    hidden = 32

    print("=== Scale generalization demo (no training, random init) ===")
    print(f"Device: {device}")
    print(f"Design-time graph size N={n_train} (models built for this N)")
    print()

    gnn = CategoryInformedGNN(feat_dim, hidden, feat_dim).to(device)
    mlp = NaiveMLP(n_train, feat_dim, hidden).to(device)

    print("(Training phase — skipped; weights are random)")
    print()

    print(f"Inference on unseen scale: N={n_test} (5× larger chain)")
    x50 = random_node_features(n_test, feat_dim, device)
    ei50 = chain_edge_index(n_test, device)

    gnn.eval()
    with torch.no_grad():
        out_gnn = gnn(x50, ei50)
    assert out_gnn.shape == (n_test, feat_dim)
    print(f"[Success] CategoryInformedGNN output shape {tuple(out_gnn.shape)} — any N is OK.")
    print()

    print("NaiveMLP forward with N=50 (expects flattened size %d)..." % (n_train * feat_dim))
    mlp.eval()
    try:
        with torch.no_grad():
            _ = mlp(x50)
    except RuntimeError as e:
        red = "\033[91m"
        reset = "\033[0m"
        print(f"{red}[Fatal Error] MLP Crashed due to size mismatch!{reset}")
        print(f"{red}{e}{reset}")
    else:
        raise RuntimeError("Expected MLP to fail with size mismatch; it did not.")


if __name__ == "__main__":
    main()
