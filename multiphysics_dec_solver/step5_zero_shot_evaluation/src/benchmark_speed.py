#!/usr/bin/env python3
"""
Benchmark forward-pass latency for the Step 4 HeteroGNN on a given ``HeteroData`` ``.pt``.

Warm-up iterations are excluded from statistics; reported times are in milliseconds.

Usage::

    python src/benchmark_speed.py --data-path PATH/to/graph.pt --model-path PATH/to/model.pth
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
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


def load_model_and_data(
    data_path: Path, model_path: Path, device: torch.device
) -> tuple[PhysicsInformedHeteroGNN, object]:
    blob = torch.load(data_path, map_location=device, weights_only=False)
    data = blob["data"].to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    primal_in = int(ckpt.get("primal_in_dim", data["primal"].x.size(-1)))
    dual_in = int(ckpt.get("dual_in_dim", data["dual"].x.size(-1)))
    if data["primal"].x.size(-1) != primal_in or data["dual"].x.size(-1) != dual_in:
        raise ValueError(
            "Feature widths do not match the checkpoint. "
            f"Data primal={data['primal'].x.size(-1)}, dual={data['dual'].x.size(-1)}; "
            f"ckpt primal={primal_in}, dual={dual_in}."
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
    return model, data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark HeteroGNN inference latency")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--warmup", type=int, default=10, help="Warm-up forwards (not timed)")
    p.add_argument("--runs", type=int, default=100, help="Timed forward passes")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_cuda = device.type == "cuda"

    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)

    model, data = load_model_and_data(args.data_path, args.model_path, device)
    x_dict = {"primal": data["primal"].x, "dual": data["dual"].x}

    def forward_once() -> None:
        with torch.no_grad():
            _ = model(data, x_dict)

    with torch.no_grad():
        for _ in range(args.warmup):
            forward_once()
            if use_cuda:
                torch.cuda.synchronize()

        timings_ms: list[float] = []
        for _ in range(args.runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward_once()
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(timings_ms)
    min_ms = min(timings_ms)
    max_ms = max(timings_ms)
    std_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0

    rows = [
        ["Device", str(device)],
        ["Warm-up forwards (excluded)", str(args.warmup)],
        ["Timed runs", str(args.runs)],
        ["Mean latency (ms)", f"{mean_ms:.4f}"],
        ["Std dev (ms)", f"{std_ms:.4f}"],
        ["Min (ms)", f"{min_ms:.4f}"],
        ["Max (ms)", f"{max_ms:.4f}"],
    ]
    print(tabulate(rows, headers=["Benchmark", "Value"], tablefmt="heavy_outline"))


if __name__ == "__main__":
    main()
