"""
spring_mass_chain_5.json を読み込み、2 層 GCN でノード回帰（MSE）のダミー学習を行う。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from import_catlab_json_to_pyg import catlab_json_to_data


class TwoLayerGCN(nn.Module):
    """入力 2 → 隠れ 16 → 出力 2（ノードごと）。"""

    def __init__(self, in_channels: int = 2, hidden_channels: int = 16, out_channels: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def main() -> None:
    root = Path(__file__).resolve().parent
    json_path = root / "spring_mass_chain_5.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"見つかりません: {json_path}\n"
            "先に `julia --project=. spring_mass_chain_export.jl` を実行してください。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = catlab_json_to_data(json_path)
    if data.x is None:
        raise ValueError("JSON にノード特徴 x がありません。")
    data = data.to(device)

    if getattr(data, "y", None) is not None:
        target = data.y
    else:
        torch.manual_seed(7)
        # JSON に y が無い場合のフォールバック（ダミー教師）
        target = data.x + 0.05 * torch.randn_like(data.x)

    model = TwoLayerGCN(in_channels=2, hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.MSELoss()

    n_epochs = 100
    print("epoch\tloss")
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"{epoch}\t{loss.item():.6f}")


if __name__ == "__main__":
    main()
