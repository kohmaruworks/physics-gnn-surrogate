#!/usr/bin/env python3
"""
Zero-shot style evaluation: load any Step-3-style ``HeteroData`` ``.pt`` and a Step 4
checkpoint, report primal-feature MSE/MAE, and save a GT vs prediction scatter figure.

Prerequisite: checkpoint ``primal_in_dim`` / ``dual_in_dim`` must match the graph.

Usage::

    python src/evaluate_generalization.py \\
        --data-path PATH/to/graph.pt \\
        --model-path PATH/to/hetero_gnn_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent.parent
STEP4_SRC = ROOT.parent / "step4_hetero_gnn_training" / "src"
STEP3_SRC = ROOT.parent / "step3_pyg_heterodata_loading" / "src"
sys.path.insert(0, str(STEP3_SRC))
sys.path.insert(0, str(STEP4_SRC))

from model import PhysicsInformedHeteroGNN, augment_reverse_edges  # noqa: E402

DEFAULT_DATA = (
    ROOT.parent
    / "step3_pyg_heterodata_loading"
    / "data"
    / "processed"
    / "hetero_cylinder_wake_t0.35.pt"
)
DEFAULT_MODEL = ROOT.parent / "step4_hetero_gnn_training" / "checkpoints" / "hetero_gnn_model.pth"
DEFAULT_OUT = ROOT / "evaluation_results" / "zeroshot_comparison.png"


def velocity_magnitude(x_uv: torch.Tensor) -> torch.Tensor:
    u = x_uv[..., 0]
    v = x_uv[..., 1]
    return torch.sqrt(u * u + v * v)


def load_model_and_data(
    data_path: Path, model_path: Path, device: torch.device
) -> tuple[PhysicsInformedHeteroGNN, Any, Any]:
    blob = torch.load(data_path, map_location=device, weights_only=False)
    data = blob["data"].to(device)
    meta = blob["meta"]

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    primal_in = int(ckpt.get("primal_in_dim", data["primal"].x.size(-1)))
    dual_in = int(ckpt.get("dual_in_dim", data["dual"].x.size(-1)))
    if data["primal"].x.size(-1) != primal_in or data["dual"].x.size(-1) != dual_in:
        raise ValueError(
            "Feature widths do not match the checkpoint (zero-shot requires same "
            f"primal/dual channel counts). Data: primal={data['primal'].x.size(-1)}, "
            f"dual={data['dual'].x.size(-1)}; checkpoint: primal={primal_in}, dual={dual_in}."
        )

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
    return model, data, meta


def plot_spatial_comparison(
    pos_xy: np.ndarray,
    mag_gt: np.ndarray,
    mag_pr: np.ndarray,
    err: np.ndarray,
    out_path: Path,
) -> None:
    vmin = float(min(mag_gt.min(), mag_pr.min()))
    vmax = float(max(mag_gt.max(), mag_pr.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300, constrained_layout=True)
    kw = {"s": 12.0, "linewidths": 0.0, "marker": "o"}

    sc0 = axes[0].scatter(
        pos_xy[:, 0], pos_xy[:, 1], c=mag_gt, cmap="turbo", vmin=vmin, vmax=vmax, **kw
    )
    axes[0].set_title("Ground truth ||u||")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(
        pos_xy[:, 0], pos_xy[:, 1], c=mag_pr, cmap="turbo", vmin=vmin, vmax=vmax, **kw
    )
    axes[1].set_title("Prediction ||u||")
    axes[1].set_aspect("equal", adjustable="box")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(pos_xy[:, 0], pos_xy[:, 1], c=err, cmap="Reds", **kw)
    axes[2].set_title("Absolute error |GT - Pred|")
    axes[2].set_aspect("equal", adjustable="box")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Zero-shot evaluation: primal velocity magnitude (fluid vertices)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HeteroGNN generalization on a HeteroData .pt")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA, help="Path to HeteroData .pt")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Step 4 checkpoint .pth")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="PNG path for scatter figure")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)

    model, data, meta = load_model_and_data(args.data_path, args.model_path, device)
    gt = data["primal"].x
    x_dict = {"primal": data["primal"].x, "dual": data["dual"].x}

    with torch.no_grad():
        pred = model(data, x_dict)

    mse = F.mse_loss(pred, gt).item()
    mae = (pred - gt).abs().mean().item()

    rows = [["MSE (primal x, all nodes)", f"{mse:.8e}"], ["MAE (primal x, all nodes)", f"{mae:.8e}"]]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    nv = meta.n_primal_vertices
    pos_v = data["primal"].pos[:nv, :2].detach().cpu().numpy()
    gt_v = gt[:nv].detach()
    pred_v = pred[:nv].detach()
    mag_gt = velocity_magnitude(gt_v).cpu().numpy()
    mag_pr = velocity_magnitude(pred_v).cpu().numpy()
    err = np.abs(mag_gt - mag_pr)

    plot_spatial_comparison(pos_v, mag_gt, mag_pr, err, args.output)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
