#!/usr/bin/env python3
"""
Validate & visualize Julia `categorical_physics_engine_step1_v1` JSON (and a simplified article schema)
with unstructured triangular meshes — PNG frames + optional GIF for Zenn.

Julia contract keys used here:
  topology.vertex_xyz, topology.triangles_flat_1based,
  node_feature_names, node_features_time_series, time

Simplified schema (examples):
  nodes [[x,y],...], triangles [[i,j,k],...],
  velocity_u / velocity_v, pressure, temperature (per-frame arrays or nested series).

GIF export (``--gif-dir``, ``--gif-from-contract``) uses PillowWriter; install Pillow if missing::

    pip install pillow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402
from tqdm import tqdm  # noqa: E402


def _configure_matplotlib_zenn_style(cmap_default: str = "turbo") -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.25,
        }
    )
    # CFD-friendly default; fallback if `turbo` unavailable.
    try:
        mpl.colormaps[cmap_default]
    except (KeyError, ValueError):
        cmap_default = "viridis"
    plt.rcParams["image.cmap"] = cmap_default


def _safe_colormap(name: str):
    try:
        return mpl.colormaps[name]
    except (KeyError, ValueError):
        return mpl.colormaps["viridis"]


def _as_float_pair_rows(vertex_xyz: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vertex_xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("vertex_xyz / nodes must be N×2 or N×3 row-wise coordinates")
    x = arr[:, 0]
    y = arr[:, 1]
    return x, y


def _triangles_from_flat_1based(flat: list[int] | np.ndarray) -> np.ndarray:
    t = np.asarray(flat, dtype=np.int64).reshape(-1, 3)
    # Julia export is 1-based
    return t - 1


def _maybe_adjust_indices(tris: np.ndarray, n_vertices: int) -> np.ndarray:
    """If triangles look 1-based (min>=1 and max==n), subtract one."""
    if tris.size == 0:
        return tris
    tmin = int(tris.min())
    tmax = int(tris.max())
    if tmin >= 1 and tmax == n_vertices:
        return tris - 1
    return tris


def extract_triangulation_from_payload(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns x, y, triangles_int (Ntri×3, 0-based)."""
    # --- Julia contract branch ---
    topo = data.get("topology")
    if isinstance(topo, dict) and "triangles_flat_1based" in topo:
        x, y = _as_float_pair_rows(topo["vertex_xyz"])
        tris = _triangles_from_flat_1based(topo["triangles_flat_1based"])
        return x, y, tris

    # --- Simplified article schema ---
    if "nodes" in data and "triangles" in data:
        x, y = _as_float_pair_rows(data["nodes"])
        tris = np.asarray(data["triangles"], dtype=np.int64)
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError("triangles must be K×3 vertex indices")
        tris = _maybe_adjust_indices(tris, len(x))
        return x, y, tris

    raise KeyError(
        "Cannot build mesh: expected Julia keys topology.triangles_flat_1based + topology.vertex_xyz "
        "or simplified keys nodes + triangles."
    )


def _julia_feature_layout(data: dict[str, Any]) -> dict[str, int]:
    names = data.get("node_feature_names")
    if not isinstance(names, list):
        return {}
    return {str(n): i for i, n in enumerate(names)}


def scalar_field_from_payload(
    data: dict[str, Any],
    quantity: str,
    frame_index: int = 0,
) -> tuple[np.ndarray, str]:
    """
    quantity: 'velocity_mag' | 'pressure' | 'temperature' | 'vx' | 'vy'
    Returns (values_per_node, colorbar_label).
    """
    layout = _julia_feature_layout(data)
    series = data.get("node_features_time_series")

    # --- Julia time-series contract ---
    if isinstance(series, list) and layout:
        if frame_index < 0:
            frame_index += len(series)
        if not (0 <= frame_index < len(series)):
            raise IndexError(f"frame_index {frame_index} out of range for {len(series)} frames")
        frame = np.asarray(series[frame_index], dtype=float)
        if frame.ndim != 2 or frame.shape[1] < len(layout):
            raise ValueError("Malformed node_features_time_series frame")

        def pick(name: str) -> np.ndarray:
            j = layout.get(name)
            if j is None:
                raise KeyError(name)
            return frame[:, j]

        if quantity == "velocity_mag":
            vx = pick("vx")
            vy = pick("vy")
            return np.hypot(vx, vy), "|u| (velocity magnitude)"
        if quantity == "vx":
            return pick("vx"), r"$u_x$"
        if quantity == "vy":
            return pick("vy"), r"$u_y$"
        if quantity == "pressure":
            return pick("p"), r"$p$"
        if quantity == "temperature":
            return pick("T"), r"$T$"
        raise ValueError(f"Unknown quantity for Julia layout: {quantity}")

    # --- Flat simplified arrays (single timestep in one JSON) ---
    def vec(key: str) -> np.ndarray | None:
        if key not in data:
            return None
        return np.asarray(data[key], dtype=float).ravel()

    vx = vec("velocity_u") if vec("velocity_u") is not None else vec("vx")
    vy = vec("velocity_v") if vec("velocity_v") is not None else vec("vy")
    pr = vec("pressure")
    T = vec("temperature")

    if quantity == "velocity_mag":
        if vx is None or vy is None:
            raise KeyError("velocity_mag requires velocity_u/velocity_v or vx/vy")
        return np.hypot(vx, vy), "|u| (velocity magnitude)"
    if quantity == "vx":
        if vx is None:
            raise KeyError("vx / velocity_u missing")
        return vx, r"$u_x$"
    if quantity == "vy":
        if vy is None:
            raise KeyError("vy / velocity_v missing")
        return vy, r"$u_y$"
    if quantity == "pressure":
        if pr is None:
            raise KeyError("pressure missing")
        return pr, r"$p$"
    if quantity == "temperature":
        if T is None:
            raise KeyError("temperature missing")
        return T, r"$T$"

    raise ValueError(
        "Could not resolve scalar field: need Julia keys `node_feature_names` + "
        "`node_features_time_series`, or flat arrays `velocity_u`/`velocity_v` (or `vx`/`vy`), "
        "`pressure`, `temperature`."
    )


def _domain_xy_limits(data: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed axes for GIF/PNG: prefer Julia contract ``mesh_lx`` / ``mesh_ly`` (physical domain), else vertex bbox."""
    lx = data.get("mesh_lx")
    ly = data.get("mesh_ly")
    if isinstance(lx, (int, float)) and isinstance(ly, (int, float)) and float(lx) > 0 and float(ly) > 0:
        return (0.0, float(lx)), (0.0, float(ly))
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-12)
    pad = 0.02 * span
    return (float(np.min(x)) - pad, float(np.max(x)) + pad), (float(np.min(y)) - pad, float(np.max(y)) + pad)


def _contour_level_bounds(vmin: float, vmax: float, n_intervals: int) -> np.ndarray:
    """Uniform contour boundaries so every frame shares the same color–value mapping."""
    return np.linspace(vmin, vmax, n_intervals + 1)


def _domain_dx_dy(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    dx = max(xlim[1] - xlim[0], 1e-12)
    dy = max(ylim[1] - ylim[0], 1e-12)
    return dx, dy


def _figsize_for_domain(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    base_height: float = 6.5,
    max_width: float = 22.0,
    extra_width_for_labels_cbar: float = 2.15,
) -> tuple[float, float]:
    """
    Wide domains (e.g. lx / ly ≈ 2) need a wide Figure; otherwise ``aspect='equal'`` shrinks the
    drawable patch into a thin horizontal strip inside a nearly square canvas.
    """
    dx, dy = _domain_dx_dy(xlim, ylim)
    plot_aspect = dx / dy
    w = min(max_width, base_height * plot_aspect + extra_width_for_labels_cbar)
    return (w, base_height)


def _apply_equal_domain_aspect(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Lock data limits and set Axes box aspect (height/width = dy/dx) so the field fills the frame."""
    dx, dy = _domain_dx_dy(xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(dy / dx)


def render_scalar_tri_contourf(
    data: dict[str, Any],
    *,
    quantity: str,
    frame_index: int,
    out_png: Path,
    title: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    x, y, tris = extract_triangulation_from_payload(data)
    vals, cbar_label = scalar_field_from_payload(data, quantity, frame_index=frame_index)

    if vals.shape[0] != x.shape[0]:
        raise ValueError(f"Field length {vals.shape[0]} does not match {x.shape[0]} vertices")

    cmap_obj = _safe_colormap(cmap or plt.rcParams["image.cmap"])

    tri = Triangulation(x, y, tris)
    xlim, ylim = _domain_xy_limits(data, x, y)
    fs = figsize if figsize is not None else _figsize_for_domain(xlim, ylim)
    fig, ax = plt.subplots(figsize=fs, constrained_layout=True)
    n_intervals = 48
    vb = _contour_level_bounds(float(np.nanmin(vals)), float(np.nanmax(vals)), n_intervals)
    norm = Normalize(vmin=vb[0], vmax=vb[-1])
    cf = ax.tricontourf(tri, vals, levels=vb, cmap=cmap_obj, norm=norm)
    _apply_equal_domain_aspect(ax, xlim, ylim)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ttl = title or f"Categorical CFD: {quantity.replace('_', ' ').title()}"
    ax.set_title(ttl)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def _natural_sort_paths(paths: list[Path]) -> list[Path]:
    def key_fn(p: Path):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", p.stem)
        return tuple(float(n) if "." in n else int(n) for n in nums) + (p.name,)

    return sorted(paths, key=key_fn)


def gif_from_json_sequence(
    json_paths: list[Path],
    *,
    quantity: str,
    out_gif: Path,
    fps: float = 8.0,
    cmap: str | None = None,
    title_template: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Each JSON file = one timestep (simplified schema). Julia multi-frame JSON: use ``gif_from_contract_timeseries``."""
    paths = _natural_sort_paths([Path(p) for p in json_paths])
    if not paths:
        raise ValueError("No JSON paths provided for GIF")

    first = json.loads(paths[0].read_text(encoding="utf-8"))
    x, y, tris = extract_triangulation_from_payload(first)
    cmap_obj = _safe_colormap(cmap or plt.rcParams["image.cmap"])

    all_vals: list[np.ndarray] = []
    _, cbar_label = scalar_field_from_payload(first, quantity, frame_index=0)
    for p in tqdm(paths, desc="GIF: load JSON frames"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        xv, yv, tv = extract_triangulation_from_payload(payload)
        if not (np.array_equal(xv, x) and np.array_equal(yv, y) and np.array_equal(tv, tris)):
            raise ValueError(f"Mesh mismatch: {p}")
        vals, _ = scalar_field_from_payload(payload, quantity, frame_index=0)
        all_vals.append(vals)

    vmin = float(min(float(np.nanmin(v)) for v in all_vals))
    vmax = float(max(float(np.nanmax(v)) for v in all_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    n_intervals = 48
    level_bounds = _contour_level_bounds(vmin, vmax, n_intervals)
    norm = Normalize(vmin=vmin, vmax=vmax)
    xlim, ylim = _domain_xy_limits(first, x, y)
    fs = figsize if figsize is not None else _figsize_for_domain(xlim, ylim)
    fig, ax = plt.subplots(figsize=fs)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.11, top=0.90)
    cbar_ref: list[Any] = [None]

    def animate(i: int):
        if cbar_ref[0] is not None:
            try:
                cbar_ref[0].remove()
            except Exception:
                pass
            cbar_ref[0] = None
        ax.clear()
        vals = all_vals[i]
        tri = Triangulation(x, y, tris)
        cf = ax.tricontourf(tri, vals, levels=level_bounds, cmap=cmap_obj, norm=norm)
        _apply_equal_domain_aspect(ax, xlim, ylim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ttl = title_template or "Categorical CFD — "
        ax.set_title(f"{ttl}{paths[i].stem}")
        cbar_ref[0] = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar_ref[0].set_label(cbar_label)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(paths), interval=1000.0 / fps, blit=False)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    except ImportError as e:
        raise RuntimeError(
            "GIF export requires Pillow. Install with: pip install pillow"
        ) from e
    plt.close(fig)


def gif_from_contract_timeseries(
    json_path: Path,
    *,
    quantity: str,
    out_gif: Path,
    fps: float = 8.0,
    cmap: str | None = None,
    title_prefix: str = "Categorical CFD",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Use Julia single-file contract with `node_features_time_series`."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    series = data.get("node_features_time_series")
    times = data.get("time")
    if not isinstance(series, list) or len(series) == 0:
        raise ValueError("Julia multi-frame GIF expects node_features_time_series list")

    x, y, tris = extract_triangulation_from_payload(data)
    cmap_obj = _safe_colormap(cmap or plt.rcParams["image.cmap"])
    n_frames = len(series)

    all_vals: list[np.ndarray] = []
    _, cbar_label = scalar_field_from_payload(data, quantity, frame_index=0)
    for i in tqdm(range(n_frames), desc="GIF: sample scalar range"):
        vals, _ = scalar_field_from_payload(data, quantity, frame_index=i)
        all_vals.append(vals)

    vmin = float(min(float(np.nanmin(v)) for v in all_vals))
    vmax = float(max(float(np.nanmax(v)) for v in all_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    n_intervals = 48
    level_bounds = _contour_level_bounds(vmin, vmax, n_intervals)
    norm = Normalize(vmin=vmin, vmax=vmax)
    xlim, ylim = _domain_xy_limits(data, x, y)
    fs = figsize if figsize is not None else _figsize_for_domain(xlim, ylim)
    fig, ax = plt.subplots(figsize=fs)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.11, top=0.90)
    cbar_ref: list[Any] = [None]

    def animate(i: int):
        if cbar_ref[0] is not None:
            try:
                cbar_ref[0].remove()
            except Exception:
                pass
            cbar_ref[0] = None
        ax.clear()
        vals = all_vals[i]
        tt = times[i] if isinstance(times, list) and i < len(times) else i
        tri = Triangulation(x, y, tris)
        cf = ax.tricontourf(tri, vals, levels=level_bounds, cmap=cmap_obj, norm=norm)
        _apply_equal_domain_aspect(ax, xlim, ylim)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(f"{title_prefix} — t = {tt}")
        cbar_ref[0] = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar_ref[0].set_label(cbar_label)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000.0 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    except ImportError as e:
        raise RuntimeError("GIF export requires Pillow (`pip install pillow`).") from e
    plt.close(fig)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="JSON contract visualization for Zenn / Py↔Julia validation.")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to JSON ground truth (Julia contract or simplified schema).",
    )
    parser.add_argument(
        "--quantity",
        choices=("velocity_mag", "pressure", "temperature", "vx", "vy"),
        default="velocity_mag",
    )
    parser.add_argument("--frame", type=int, default=-1, help="Frame index for Julia time series (negative = last).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--cmap", default="turbo", help="Colormap name (falls back to viridis).")
    parser.add_argument("--gif-dir", type=Path, default=None, help="Directory of per-timestep JSON files for GIF.")
    parser.add_argument("--gif-out", type=Path, default=None, help="Output GIF path.")
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=8.0,
        help="GIF playback speed (lower = slower panning). Default 8.",
    )
    parser.add_argument(
        "--gif-from-contract",
        action="store_true",
        help="Build GIF from --json using node_features_time_series (Julia single-file export).",
    )
    args = parser.parse_args()

    _configure_matplotlib_zenn_style(cmap_default=args.cmap)

    repo_assets = Path(__file__).resolve().parent.parent / "zenn_assets"
    default_json = Path(__file__).resolve().parent.parent / "data" / "raw" / "ground_truth_cylinder_wake.json"

    jpath = args.json if args.json is not None else default_json

    if args.gif_dir is not None:
        jp = _natural_sort_paths(list(Path(args.gif_dir).glob("*.json")))
        out_g = args.gif_out or (repo_assets / "timeseries_from_dir.gif")
        gif_from_json_sequence(jp, quantity=args.quantity, out_gif=out_g, fps=args.gif_fps, cmap=args.cmap)
        print(f"Wrote GIF: {out_g}")
        return

    if args.gif_from_contract:
        out_g = args.gif_out or (repo_assets / "timeseries_contract.gif")
        gif_from_contract_timeseries(jpath, quantity=args.quantity, out_gif=out_g, fps=args.gif_fps, cmap=args.cmap)
        print(f"Wrote GIF: {out_g}")
        return

    out_png = args.out or (repo_assets / f"contract_{args.quantity}_frame{args.frame}.png")
    data = json.loads(Path(jpath).read_text(encoding="utf-8"))
    render_scalar_tri_contourf(
        data,
        quantity=args.quantity,
        frame_index=args.frame,
        out_png=out_png,
        cmap=args.cmap,
        title=f"Categorical CFD: {args.quantity.replace('_', ' ').title()}",
    )
    print(f"Wrote PNG: {out_png}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        # Default smoke test: render last timestep when Julia export is present.
        _configure_matplotlib_zenn_style()
        _here = Path(__file__).resolve().parent
        _default_contract = _here.parent / "data" / "raw" / "ground_truth_cylinder_wake.json"
        _out_dir = _here.parent / "zenn_assets"
        _out_dir.mkdir(parents=True, exist_ok=True)

        if _default_contract.is_file():
            _data = json.loads(_default_contract.read_text(encoding="utf-8"))
            try:
                render_scalar_tri_contourf(
                    _data,
                    quantity="velocity_mag",
                    frame_index=-1,
                    out_png=_out_dir / "contract_velocity_mag_sample.png",
                    title="Categorical CFD: Velocity Field (JSON contract)",
                )
                print(f"Sample PNG written: {_out_dir / 'contract_velocity_mag_sample.png'}")
            except Exception as exc:
                print(f"Sample render skipped ({exc}). Run Julia Step 1 or pass --json explicitly.")
        else:
            print(
                f"No sample JSON at {_default_contract}; pass CLI flags, e.g.\n"
                f"  python {Path(__file__).name} --json /path/to/ground_truth.json --quantity velocity_mag"
            )
