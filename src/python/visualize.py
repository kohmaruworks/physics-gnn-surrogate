"""
ロールアウト予測とグラウンドトゥルースの比較可視化（matplotlib）。

スカラー場（p / T 等）を tripcolor、流速 (vx, vy) を quiver で重ね描き。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

PYDIR = Path(__file__).resolve().parent
if str(PYDIR) not in sys.path:
    sys.path.insert(0, str(PYDIR))

from dataset import default_interim_path, load_interim_json


def _project_root() -> Path:
    return PYDIR.parent.parent


def infer_node_positions_2d(raw: Dict[str, Any]) -> np.ndarray:
    """JSON に座標が無い場合: 既知 toy（4 ノード・単位正方形）にフォールバック。"""
    if "node_positions" in raw:
        return np.asarray(raw["node_positions"], dtype=np.float64)
    n = int(raw["num_nodes"])
    if n == 4:
        return np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(n, 2))


def _square_axis_limits(pos: np.ndarray, pad: float = 0.15) -> Tuple[float, float, float, float]:
    xmin, xmax = float(pos[:, 0].min()), float(pos[:, 0].max())
    ymin, ymax = float(pos[:, 1].min()), float(pos[:, 1].max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = max(xmax - xmin, ymax - ymin) * 0.5 + pad
    return cx - half, cx + half, cy - half, cy + half


def edges_from_topology_json(raw: Dict[str, Any]) -> List[Tuple[int, int]]:
    top = raw["topology"]
    ei = np.asarray(top["edge_index"], dtype=np.int64).reshape(2, -1)
    seen = set()
    out: List[Tuple[int, int]] = []
    for c in range(ei.shape[1]):
        a, b = int(ei[0, c]), int(ei[1, c])
        if a == b:
            continue
        e = (a, b) if a < b else (b, a)
        if e not in seen:
            seen.add(e)
            out.append((a, b))
    return out


def triangulation_from_raw(raw: Dict[str, Any], pos: np.ndarray) -> Optional[Triangulation]:
    top = raw.get("topology", {})
    if "triangles" not in top or top["triangles"] is None:
        return None
    flat = np.asarray(top["triangles"], dtype=np.int64)
    if flat.size == 0 or flat.size % 3 != 0:
        return None
    tri = flat.reshape(-1, 3)
    if int(tri.max()) >= pos.shape[0]:
        return None
    return Triangulation(pos[:, 0].copy(), pos[:, 1].copy(), triangles=tri.copy())


def feature_scalar_index(raw: Dict[str, Any], scalar_name: str) -> int:
    names = list(raw.get("node_feature_names", ["vx", "vy", "p", "T"]))
    lower = [str(x).lower() for x in names]
    key = scalar_name.strip().lower()
    if key in lower:
        return lower.index(key)
    if key == "temperature" and "t" in lower:
        return lower.index("t")
    if key == "pressure" and "p" in lower:
        return lower.index("p")
    return 3


def mse_all_features(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((gt - pred) ** 2))


def _draw_mesh_outline(ax: Axes, pos: np.ndarray, edges: List[Tuple[int, int]], *, color: str = "0.35", lw: float = 0.7) -> None:
    if not edges:
        return
    segs = np.array([[[pos[a, 0], pos[a, 1]], [pos[b, 0], pos[b, 1]]] for a, b in edges])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=5))


def _quiver_speed_scale(pos: np.ndarray, u: np.ndarray, v: np.ndarray, *, ref_len_frac: float = 0.12) -> float:
    """軸スパンに対する見やすい quiver scale（大きいほど矢印は短く見える）。"""
    span = max(
        float(pos[:, 0].max() - pos[:, 0].min()),
        float(pos[:, 1].max() - pos[:, 1].min()),
        1e-9,
    )
    spd = np.sqrt(np.clip(u * u + v * v, 0, None))
    smax = float(spd.max()) if spd.size else 1e-9
    return smax / (ref_len_frac * span + 1e-12)


def plot_rich_flow_side_by_side(
    pos: np.ndarray,
    tri: Optional[Triangulation],
    edges: List[Tuple[int, int]],
    state_gt: np.ndarray,
    state_pred: np.ndarray,
    *,
    scalar_idx: int,
    scalar_label: str,
    title_gt: str,
    title_pred: str,
    outfile: Path,
    suptitle: str = "",
    scalar_vmin_vmax: Optional[Tuple[float, float]] = None,
    speed_vmin_vmax: Optional[Tuple[float, float]] = None,
) -> None:
    vx, vy = 0, 1
    mse = mse_all_features(state_gt, state_pred)
    sc_gt = state_gt[:, scalar_idx].astype(np.float64)
    sc_pr = state_pred[:, scalar_idx].astype(np.float64)
    if scalar_vmin_vmax is None:
        smin = float(min(sc_gt.min(), sc_pr.min()))
        smax = float(max(sc_gt.max(), sc_pr.max()))
        if smin == smax:
            smax = smin + 1e-9
        scalar_vmin_vmax = (smin, smax)

    u_gt, v_gt = state_gt[:, vx].astype(np.float64), state_gt[:, vy].astype(np.float64)
    u_pr, v_pr = state_pred[:, vx].astype(np.float64), state_pred[:, vy].astype(np.float64)
    sp_gt = np.sqrt(np.clip(u_gt * u_gt + v_gt * v_gt, 0, None))
    sp_pr = np.sqrt(np.clip(u_pr * u_pr + v_pr * v_pr, 0, None))
    if speed_vmin_vmax is None:
        qmin, qmax = 0.0, float(max(sp_gt.max(), sp_pr.max(), 1e-9))
        speed_vmin_vmax = (qmin, qmax)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=False)
    plt.subplots_adjust(wspace=0.35)
    x0, x1, y0, y1 = _square_axis_limits(pos)
    cnorm_s = Normalize(vmin=scalar_vmin_vmax[0], vmax=scalar_vmin_vmax[1])
    cnorm_q = Normalize(vmin=speed_vmin_vmax[0], vmax=speed_vmin_vmax[1])

    def panel(ax: Axes, sc: np.ndarray, u: np.ndarray, v: np.ndarray, sp: np.ndarray, title: str) -> None:
        ax.set_facecolor("#f2f2f2")
        if tri is not None:
            tc = ax.tripcolor(
                tri,
                sc,
                shading="gouraud",
                cmap="coolwarm",
                norm=cnorm_s,
                zorder=1,
            )
        else:
            tc = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=sc,
                cmap="coolwarm",
                norm=cnorm_s,
                s=280,
                edgecolors="k",
                linewidths=0.4,
                zorder=2,
            )
        _draw_mesh_outline(ax, pos, edges)
        scale = _quiver_speed_scale(pos, u, v)
        qv = ax.quiver(
            pos[:, 0],
            pos[:, 1],
            u,
            v,
            sp,
            cmap="viridis",
            norm=cnorm_q,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.012,
            headwidth=3.2,
            headlength=4.0,
            zorder=6,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        c1 = fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.02, shrink=0.82)
        c1.set_label(scalar_label)
        c2 = fig.colorbar(qv, ax=ax, fraction=0.046, pad=0.16, shrink=0.82)
        c2.set_label("|v|")
        return tc, qv

    title_gt_full = f"{title_gt}\nMSE(all channels) = {mse:.6e}"
    title_pr_full = f"{title_pred}\nMSE(all channels) = {mse:.6e}"
    panel(axes[0], sc_gt, u_gt, v_gt, sp_gt, title_gt_full)
    panel(axes[1], sc_pr, u_pr, v_pr, sp_pr, title_pr_full)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_rollout_animation_rich(
    pos: np.ndarray,
    tri: Optional[Triangulation],
    edges: List[Tuple[int, int]],
    gt_series: np.ndarray,
    pred_series: np.ndarray,
    time_values: np.ndarray,
    t_indices_gt: np.ndarray,
    outfile: Path,
    *,
    scalar_idx: int,
    scalar_label: str,
    scalar_vmin_vmax: Tuple[float, float],
    speed_vmin_vmax: Tuple[float, float],
    fps: int = 4,
) -> None:
    """GIF: 各フレームで tripcolor / scatter + quiver を再描画（全フレーム共通の色レンジ・quiver scale）。"""
    vx, vy = 0, 1
    s = pred_series.shape[0]
    cnorm_s = Normalize(vmin=scalar_vmin_vmax[0], vmax=scalar_vmin_vmax[1])
    cnorm_q = Normalize(vmin=speed_vmin_vmax[0], vmax=speed_vmin_vmax[1])

    def _max_scale_over(series: np.ndarray) -> float:
        m = 0.0
        for i in range(series.shape[0]):
            m = max(
                m,
                _quiver_speed_scale(pos, series[i, :, vx].astype(np.float64), series[i, :, vy].astype(np.float64)),
            )
        return max(m, 1e-9)

    global_scale = max(_max_scale_over(gt_series), _max_scale_over(pred_series))
    x0, x1, y0, y1 = _square_axis_limits(pos)

    def _one_panel(ax: Axes, state: np.ndarray, title: str) -> None:
        ax.set_facecolor("#f2f2f2")
        sc = state[:, scalar_idx].astype(np.float64)
        u = state[:, vx].astype(np.float64)
        v = state[:, vy].astype(np.float64)
        sp = np.sqrt(np.clip(u * u + v * v, 0, None))
        if tri is not None:
            m_s = ax.tripcolor(
                tri, sc, shading="gouraud", cmap="coolwarm", norm=cnorm_s, zorder=1
            )
        else:
            m_s = ax.scatter(
                pos[:, 0], pos[:, 1], c=sc, cmap="coolwarm", norm=cnorm_s, s=280, ec="k", lw=0.4, zorder=2
            )
        _draw_mesh_outline(ax, pos, edges)
        qv = ax.quiver(
            pos[:, 0],
            pos[:, 1],
            u,
            v,
            sp,
            cmap="viridis",
            norm=cnorm_q,
            angles="xy",
            scale_units="xy",
            scale=global_scale,
            width=0.012,
            headwidth=3.2,
            headlength=4.0,
            zorder=6,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(m_s, ax=ax, fraction=0.046, pad=0.02, shrink=0.82).set_label(scalar_label)
        fig.colorbar(qv, ax=ax, fraction=0.046, pad=0.16, shrink=0.82).set_label("|v|")

    fig = plt.figure(figsize=(12.5, 5.2))

    def frame_fn(frame: int) -> tuple:
        fig.clf()
        axes = fig.subplots(1, 2)
        plt.subplots_adjust(wspace=0.35)
        vg = gt_series[frame]
        vp = pred_series[frame]
        idx = int(t_indices_gt[frame])
        mse = mse_all_features(vg, vp)
        _one_panel(
            axes[0],
            vg,
            f"Ground truth  t={time_values[idx]:.4f}\nMSE(all channels) = {mse:.6e}",
        )
        _one_panel(
            axes[1],
            vp,
            f"GNN prediction  step {frame + 1}/{s}\nMSE(all channels) = {mse:.6e}",
        )
        fig.suptitle(f"Scalar ({scalar_label}) + velocity quiver", fontsize=12, y=1.02)
        return ()

    anim = animation.FuncAnimation(fig, frame_fn, frames=s, blit=False)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    anim.save(outfile, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def _global_scalar_speed_limits(
    gt_series: np.ndarray, pred_series: np.ndarray, scalar_idx: int
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    vx, vy = 0, 1
    sc = np.concatenate(
        [
            gt_series[:, :, scalar_idx].ravel(),
            pred_series[:, :, scalar_idx].ravel(),
        ]
    )
    smin, smax = float(sc.min()), float(sc.max())
    if smin == smax:
        smax = smin + 1e-9
    spd = np.sqrt(
        np.concatenate(
            [
                gt_series[:, :, vx].ravel() ** 2 + gt_series[:, :, vy].ravel() ** 2,
                pred_series[:, :, vx].ravel() ** 2 + pred_series[:, :, vy].ravel() ** 2,
            ]
        )
    )
    qmax = float(spd.max()) if spd.size else 1e-9
    return (smin, smax), (0.0, qmax)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rich flow viz: scalar field + quiver")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--pred-npy", type=str, default="", dest="pred_npy")
    ap.add_argument("--meta-npz", type=str, default="", dest="meta_npz")
    ap.add_argument("--step", type=int, default=-1, help="-1: ロールアウト最終ステップ")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--scalar", type=str, default="T", help="背景スカラー場: T / p など node_feature_names に合わせる")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    root = _project_root()
    json_path = Path(args.json) if args.json else default_interim_path(root)
    raw = load_interim_json(json_path)
    nfs = np.asarray(raw["node_features_time_series"], dtype=np.float32)
    pred_path = Path(args.pred_npy) if args.pred_npy else root / "data" / "interim" / "rollout_predictions.npy"
    primal_pred = np.load(pred_path)
    meta_path = Path(args.meta_npz) if args.meta_npz else root / "data" / "interim" / "rollout_meta.npz"
    t_axis = np.asarray(raw["time"], dtype=np.float64)
    if meta_path.is_file():
        meta = np.load(meta_path)
        t_start = int(meta["t_start"])
        h = int(meta["history_len"])
        s_roll = int(meta["rollout_steps"])
        pred_t0 = int(meta["pred_time_index_start"])
    else:
        t_start, h = 0, 1
        s_roll = int(primal_pred.shape[0])
        pred_t0 = h
        print("警告: rollout_meta.npz がありません。t_start=0, history_len=1 を仮定します。")

    scalar_idx = feature_scalar_index(raw, args.scalar)
    names = list(raw.get("node_feature_names", ["vx", "vy", "p", "T"]))
    scalar_label = names[scalar_idx] if scalar_idx < len(names) else args.scalar

    step = args.step if args.step >= 0 else primal_pred.shape[0] - 1
    step = max(0, min(step, primal_pred.shape[0] - 1))

    gt_time_idx = t_start + h + step
    v_gt = nfs[gt_time_idx]
    v_pred = primal_pred[step]

    pos = infer_node_positions_2d(raw)
    edges = edges_from_topology_json(raw)
    tri = triangulation_from_raw(raw, pos)

    t_indices = pred_t0 + np.arange(s_roll, dtype=np.int32)
    t_indices = np.clip(t_indices, 0, len(t_axis) - 1)
    gt_aligned = np.stack([nfs[int(i)] for i in t_indices], axis=0)

    lim_s, lim_q = _global_scalar_speed_limits(gt_aligned, primal_pred, scalar_idx)

    out_png = root / "data" / "interim" / "comparison_plot.png"
    plot_rich_flow_side_by_side(
        pos,
        tri,
        edges,
        v_gt,
        v_pred,
        scalar_idx=scalar_idx,
        scalar_label=scalar_label,
        title_gt=f"Ground truth (t = {t_axis[gt_time_idx]:.4f})",
        title_pred=f"HeteroGNN autoregressive (step {step + 1}/{s_roll})",
        outfile=out_png,
        suptitle=f"Phase 2 — scalar ({scalar_label}) + velocity quiver",
        scalar_vmin_vmax=lim_s,
        speed_vmin_vmax=lim_q,
    )
    print(f"Saved: {out_png}")

    if args.animate:
        gif_path = root / "data" / "interim" / "rollout_animation.gif"
        save_rollout_animation_rich(
            pos,
            tri,
            edges,
            gt_aligned,
            primal_pred,
            t_axis,
            t_indices,
            gif_path,
            scalar_idx=scalar_idx,
            scalar_label=scalar_label,
            scalar_vmin_vmax=lim_s,
            speed_vmin_vmax=lim_q,
            fps=args.fps,
        )
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
