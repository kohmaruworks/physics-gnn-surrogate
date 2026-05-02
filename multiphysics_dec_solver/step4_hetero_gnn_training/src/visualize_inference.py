#!/usr/bin/env python3
"""
Spatial comparison of ground-truth vs Physics-Informed HeteroGNN primal velocities.

Loads Step 3 ``hetero_cylinder_wake_t0.35.pt`` and ``checkpoints/hetero_gnn_model.pth``,
runs a forward pass, and saves a 1×3 scatter figure (velocity magnitude + absolute error).

Usage (from ``step4_hetero_gnn_training/``)::

    python src/visualize_inference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
STEP3_SRC = ROOT.parent / "step3_pyg_heterodata_loading" / "src"
sys.path.insert(0, str(STEP3_SRC))
sys.path.insert(0, str(SRC))

from model import PhysicsInformedHeteroGNN, augment_reverse_edges  # noqa: E402

DEFAULT_PT = (
    ROOT.parent
    / "step3_pyg_heterodata_loading"
    / "data"
    / "processed"
    / "hetero_cylinder_wake_t0.35.pt"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "hetero_gnn_model.pth"
OUT_DIR = ROOT / "zenn_assets"
OUT_PATH = OUT_DIR / "gnn_inference_comparison.png"


def _velocity_magnitude(x_uv: torch.Tensor) -> torch.Tensor:
    """``x_uv``: ``[..., 2]`` channels interpreted as ``(u, v)``."""
    u = x_uv[..., 0]
    v = x_uv[..., 1]
    return torch.sqrt(u * u + v * v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Step 4 HeteroGNN inference vs GT")
    p.add_argument("--data-path", type=Path, default=DEFAULT_PT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if not args.data_path.is_file():
        raise FileNotFoundError(f"Missing HeteroData tensor: {args.data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {args.checkpoint}")

    blob = torch.load(args.data_path, map_location=device, weights_only=False)
    data = blob["data"].to(device)
    meta = blob["meta"]
    nv = meta.n_primal_vertices

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    primal_in = int(ckpt.get("primal_in_dim", data["primal"].x.size(-1)))
    dual_in = int(ckpt.get("dual_in_dim", data["dual"].x.size(-1)))
    ta = ckpt.get("train_args") or {}
    if not isinstance(ta, dict):
        ta = vars(ta) if hasattr(ta, "__dict__") else {}
    hidden_dim = int(ta.get("hidden_dim", 64))
    num_layers = int(ta.get("num_layers", 2))

    model = PhysicsInformedHeteroGNN(
        primal_in_dim=primal_in,
        dual_in_dim=dual_in,
        hidden_dim=hidden_dim,
        primal_out_dim=primal_in,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = augment_reverse_edges(data)
    x_dict = {"primal": data["primal"].x, "dual": data["dual"].x}

    pos_v = data["primal"].pos[:nv, :2].detach().cpu().numpy()
    gt_x = data["primal"].x[:nv].detach()

    with torch.no_grad():
        pred_x = model(data, x_dict)[:nv].detach()

    mag_gt = _velocity_magnitude(gt_x).cpu().numpy()
    mag_pr = _velocity_magnitude(pred_x).cpu().numpy()
    err = np.abs(mag_gt - mag_pr)

    vmin = float(min(mag_gt.min(), mag_pr.min()))
    vmax = float(max(mag_gt.max(), mag_pr.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300, constrained_layout=True)
    kw_scatter = {"s": 12.0, "linewidths": 0.0, "marker": "o"}

    sc0 = axes[0].scatter(
        pos_v[:, 0], pos_v[:, 1], c=mag_gt, cmap="turbo", vmin=vmin, vmax=vmax, **kw_scatter
    )
    axes[0].set_title("Ground truth velocity magnitude ||u||")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(
        pos_v[:, 0], pos_v[:, 1], c=mag_pr, cmap="turbo", vmin=vmin, vmax=vmax, **kw_scatter
    )
    axes[1].set_title("GNN prediction velocity magnitude ||u||")
    axes[1].set_aspect("equal", adjustable="box")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(pos_v[:, 0], pos_v[:, 1], c=err, cmap="Reds", **kw_scatter)
    axes[2].set_title("Absolute error |GT - Pred|")
    axes[2].set_aspect("equal", adjustable="box")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.suptitle(
        "Physics-informed HeteroGNN: primal velocity magnitude (fluid vertices)",
        fontsize=11,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
