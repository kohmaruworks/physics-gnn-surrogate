#!/usr/bin/env python3
"""
Build a 1×3 GIF comparing ground-truth velocity magnitude, GNN prediction, and absolute error
over time using ``matplotlib.animation.FuncAnimation``.

Supports:

1. **Sequence mode** — multiple Step-3-style ``.pt`` snapshots (same mesh / topology). Each frame
   uses the ground-truth fields from that file and runs one forward pass.

2. **Rollout mode** (single ``--data-path``) — autoregressive passes: the primal input features are
   fed back from the previous prediction while dual features stay fixed (geometry). The left panel
   shows the reference ground truth from the initial snapshot; center/right show how the surrogate
   drifts when iterated.

Dimension checks align with ``evaluate_generalization.py``: primal/dual feature widths must match
the checkpoint before any ``forward`` call.

Usage (from ``step5_zero_shot_evaluation/``)::

    python src/generate_comparison_gif.py \\
        --model-path ../step4_hetero_gnn_training/checkpoints/hetero_gnn_model.pth \\
        --output evaluation_results/zeroshot_comparison_animation.gif

    # Multiple timesteps (sorted glob)
    python src/generate_comparison_gif.py \\
        --sequence-glob '../step3_pyg_heterodata_loading/data/processed/hetero_cylinder_wake_t*.pt'
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

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
DEFAULT_OUT = ROOT / "evaluation_results" / "zeroshot_comparison_animation.gif"
DEFAULT_SEQ_GLOB = (
    ROOT.parent
    / "step3_pyg_heterodata_loading"
    / "data"
    / "processed"
    / "hetero_cylinder_wake_t*.pt"
)


def velocity_magnitude(x_uv: torch.Tensor) -> torch.Tensor:
    u = x_uv[..., 0]
    v = x_uv[..., 1]
    return torch.sqrt(u * u + v * v)


def _validate_feature_dims(data, primal_in: int, dual_in: int, ctx: str) -> None:
    """Raise ``ValueError`` if node feature widths disagree with the checkpoint."""
    pd = data["primal"].x.size(-1)
    dd = data["dual"].x.size(-1)
    if pd != primal_in or dd != dual_in:
        raise ValueError(
            f"{ctx}: feature widths must match the checkpoint for stable inference. "
            f"Got primal x.size(1)={pd}, dual x.size(1)={dd}; "
            f"checkpoint expects primal={primal_in}, dual={dual_in}."
        )
    assert data["primal"].x.size(1) == primal_in
    assert data["dual"].x.size(1) == dual_in


def _assert_topology_equal(ref: object, other: object, path: Path) -> None:
    """Ensure unknown meshes do not silently reuse the wrong edge indices."""
    if ref["primal"].num_nodes != other["primal"].num_nodes:
        raise ValueError(
            f"Primal node count mismatch vs reference ({path}): "
            f"{ref['primal'].num_nodes} vs {other['primal'].num_nodes}"
        )
    if ref["dual"].num_nodes != other["dual"].num_nodes:
        raise ValueError(
            f"Dual node count mismatch vs reference ({path}): "
            f"{ref['dual'].num_nodes} vs {other['dual'].num_nodes}"
        )
    keys = [
        ("primal", "p2p", "primal"),
        ("dual", "d2d", "dual"),
        ("primal", "p2d", "dual"),
    ]
    for key in keys:
        ei_r = ref[key].edge_index
        ei_o = other[key].edge_index
        if ei_r.shape != ei_o.shape or not torch.equal(ei_r.cpu(), ei_o.cpu()):
            raise ValueError(f"edge_index mismatch for {key} when comparing to reference ({path}).")


def load_checkpoint_model(model_path: Path, device: torch.device) -> tuple[PhysicsInformedHeteroGNN, dict]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if ckpt.get("primal_in_dim") is None or ckpt.get("dual_in_dim") is None:
        raise ValueError(
            "Checkpoint must define primal_in_dim and dual_in_dim (re-save after Step 4 training)."
        )
    primal_in = int(ckpt["primal_in_dim"])
    dual_in = int(ckpt["dual_in_dim"])
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
    return model, ckpt


def load_hetero_blob(path: Path, device: torch.device) -> tuple[object, object]:
    blob = torch.load(path, map_location=device, weights_only=False)
    data = blob["data"].to(device)
    meta = blob["meta"]
    return data, meta


def collect_sequence_frames(
    model: PhysicsInformedHeteroGNN,
    ckpt: dict,
    paths: list[Path],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str]]:
    primal_in = int(ckpt["primal_in_dim"])
    dual_in = int(ckpt["dual_in_dim"])

    gt_mags: list[np.ndarray] = []
    pr_mags: list[np.ndarray] = []
    titles: list[str] = []

    ref_data = None
    pos_xy: np.ndarray | None = None

    for p in paths:
        data, meta = load_hetero_blob(p, device)
        _validate_feature_dims(data, primal_in, dual_in, ctx=f"sequence file {p.name}")

        data = augment_reverse_edges(data)
        if ref_data is None:
            ref_data = data
        else:
            _assert_topology_equal(ref_data, data, p)

        if pos_xy is None:
            nv = meta.n_primal_vertices
            pos_xy = data["primal"].pos[:nv, :2].detach().cpu().numpy()

        gt = data["primal"].x
        x_dict = {"primal": data["primal"].x, "dual": data["dual"].x}
        with torch.no_grad():
            pred = model(data, x_dict)

        nv = meta.n_primal_vertices
        gt_v = gt[:nv].detach()
        pred_v = pred[:nv].detach()
        mag_gt = velocity_magnitude(gt_v).cpu().numpy()
        mag_pr = velocity_magnitude(pred_v).cpu().numpy()
        gt_mags.append(mag_gt)
        pr_mags.append(mag_pr)
        titles.append(p.stem)

    assert pos_xy is not None
    err_mags = [np.abs(gt_mags[i] - pr_mags[i]) for i in range(len(gt_mags))]
    return gt_mags, pr_mags, err_mags, pos_xy, titles


def collect_rollout_frames(
    model: PhysicsInformedHeteroGNN,
    ckpt: dict,
    data_path: Path,
    device: torch.device,
    rollout_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str]]:
    primal_in = int(ckpt["primal_in_dim"])
    dual_in = int(ckpt["dual_in_dim"])

    data, meta = load_hetero_blob(data_path, device)
    _validate_feature_dims(data, primal_in, dual_in, ctx=f"rollout {data_path.name}")
    data = augment_reverse_edges(data)

    nv = meta.n_primal_vertices
    pos_xy = data["primal"].pos[:nv, :2].detach().cpu().numpy()

    gt0 = data["primal"].x[:nv].detach()
    mag_gt_ref = velocity_magnitude(gt0).cpu().numpy()

    gt_mags: list[np.ndarray] = []
    pr_mags: list[np.ndarray] = []
    titles: list[str] = []

    x_primal = data["primal"].x.clone()
    x_dual = data["dual"].x.clone()

    for step in range(rollout_steps):
        x_dict = {"primal": x_primal, "dual": x_dual}
        with torch.no_grad():
            pred_full = model(data, x_dict)

        pred_v = pred_full[:nv].detach()
        mag_pr = velocity_magnitude(pred_v).cpu().numpy()
        gt_mags.append(mag_gt_ref.copy())
        pr_mags.append(mag_pr)
        titles.append(f"rollout step {step}")
        x_primal = pred_full.clone()

    err_mags = [np.abs(gt_mags[i] - pr_mags[i]) for i in range(len(gt_mags))]
    return gt_mags, pr_mags, err_mags, pos_xy, titles


def build_animation(
    pos_xy: np.ndarray,
    gt_mags: list[np.ndarray],
    pr_mags: list[np.ndarray],
    err_mags: list[np.ndarray],
    frame_titles: list[str],
    suptitle: str,
    *,
    fps: float,
) -> tuple[FuncAnimation, plt.Figure]:
    vmin = float(min(m.min() for m in gt_mags + pr_mags))
    vmax = float(max(m.max() for m in gt_mags + pr_mags))
    err_max = float(max(m.max() for m in err_mags))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    kw = {"s": 14.0, "linewidths": 0.0, "marker": "o"}

    sc0 = axes[0].scatter(pos_xy[:, 0], pos_xy[:, 1], c=gt_mags[0], cmap="turbo", vmin=vmin, vmax=vmax, **kw)
    axes[0].set_title("Ground truth ‖u‖")
    axes[0].set_aspect("equal", adjustable="box")
    _ = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(pos_xy[:, 0], pos_xy[:, 1], c=pr_mags[0], cmap="turbo", vmin=vmin, vmax=vmax, **kw)
    axes[1].set_title("GNN prediction ‖u‖")
    axes[1].set_aspect("equal", adjustable="box")
    _ = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(pos_xy[:, 0], pos_xy[:, 1], c=err_mags[0], cmap="Reds", vmin=0.0, vmax=err_max, **kw)
    axes[2].set_title("Absolute error |‖u‖_GT − ‖u‖_pred|")
    axes[2].set_aspect("equal", adjustable="box")
    _ = fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(suptitle + " — " + frame_titles[0], fontsize=11)

    def _update(frame: int) -> tuple:
        sc0.set_array(gt_mags[frame])
        sc1.set_array(pr_mags[frame])
        sc2.set_array(err_mags[frame])
        fig.suptitle(suptitle + " — " + frame_titles[frame], fontsize=11)
        return sc0, sc1, sc2

    interval_ms = max(1, int(round(1000.0 / fps)))
    anim = FuncAnimation(fig, _update, frames=len(gt_mags), interval=interval_ms, blit=False)
    return anim, fig


def resolve_sequence_paths(sequence_glob_user: str | None) -> list[Path]:
    pattern = sequence_glob_user if sequence_glob_user else str(DEFAULT_SEQ_GLOB)
    hits = sorted({Path(p).resolve() for p in glob(pattern)})
    if hits:
        return hits
    if not Path(pattern).is_absolute():
        hits = sorted({Path(p).resolve() for p in glob(str(ROOT / pattern))})
    return hits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GIF: GT vs GNN velocity magnitude vs absolute error")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA,
        help="Single snapshot .pt for rollout mode or fallback when glob finds one file",
    )
    p.add_argument(
        "--sequence-glob",
        type=str,
        default=None,
        help="Glob for multiple .pt snapshots (same topology). If omitted, tries default processed glob.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=24,
        help="Frames for autoregressive rollout when only one snapshot is available",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model, ckpt = load_checkpoint_model(args.model_path, device)

    glob_pat = args.sequence_glob if args.sequence_glob is not None else str(DEFAULT_SEQ_GLOB)
    paths = resolve_sequence_paths(args.sequence_glob)

    if len(paths) >= 2:
        gt_mags, pr_mags, err_mags, pos_xy, titles = collect_sequence_frames(model, ckpt, paths, device)
        suptitle = "Zero-shot sequence (per-frame GT vs one-step inference)"
    else:
        if not args.data_path.is_file():
            raise FileNotFoundError(
                f"No sequence snapshots matched '{glob_pat}' (need ≥2), and single-file path missing: {args.data_path}"
            )
        gt_mags, pr_mags, err_mags, pos_xy, titles = collect_rollout_frames(
            model, ckpt, args.data_path, device, rollout_steps=max(2, args.rollout_steps)
        )
        suptitle = "Autoregressive rollout (reference ‖u‖ from initial snapshot)"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim, fig = build_animation(pos_xy, gt_mags, pr_mags, err_mags, titles, suptitle, fps=args.fps)
    anim.save(args.output, writer="pillow", dpi=300, fps=args.fps)
    plt.close(fig)
    print(f"Saved GIF: {args.output}")


if __name__ == "__main__":
    main()
