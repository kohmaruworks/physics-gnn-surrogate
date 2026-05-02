#!/usr/bin/env python3
"""
Overfit sanity check for Physics-Informed HeteroGNN on Step 3 ``HeteroData``.

Loads ``hetero_cylinder_wake_t0.35.pt``, reconstructs primal ``x`` while adding
a pseudo divergence-free penalty on primal vertex velocities.

Usage (from ``step4_hetero_gnn_training/``)::

    pip install -r requirements_step4.txt
    python src/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional logging dependency
    SummaryWriter = None  # type: ignore[misc, assignment]


class _NullSummaryWriter:
    def add_scalar(self, *args: object, **kwargs: object) -> None:
        return None

    def close(self) -> None:
        return None


def _make_writer(log_dir: Path):
    if SummaryWriter is None:
        tqdm.write(
            "warning: tensorboard not installed; skipping run logs "
            "(pip install tensorboard or install requirements_step4.txt)."
        )
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))  # type: ignore[return-value]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
STEP3_SRC = ROOT.parent / "step3_pyg_heterodata_loading" / "src"
# Pickled Step 3 ``HeteroData`` references ``hetero_dataset`` for unpickling.
sys.path.insert(0, str(STEP3_SRC))
sys.path.insert(0, str(SRC))

from model import PhysicsInformedHeteroGNN, augment_reverse_edges  # noqa: E402
from physics_loss import physics_informed_total_loss  # noqa: E402

DEFAULT_PT = (
    ROOT.parent
    / "step3_pyg_heterodata_loading"
    / "data"
    / "processed"
    / "hetero_cylinder_wake_t0.35.pt"
)
CHECKPOINT = ROOT / "checkpoints" / "hetero_gnn_model.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4 Physics-Informed HeteroGNN trainer")
    p.add_argument("--data-path", type=Path, default=DEFAULT_PT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lambda-physics", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-dir", type=Path, default=ROOT / "runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if not args.data_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint tensor: {args.data_path}")

    blob = torch.load(args.data_path, map_location=device, weights_only=False)
    data = blob["data"].to(device)
    meta = blob["meta"]
    data = augment_reverse_edges(data)

    x_dict = {"primal": data["primal"].x, "dual": data["dual"].x}
    target_primal = data["primal"].x.clone()

    primal_in = data["primal"].x.size(-1)
    dual_in = data["dual"].x.size(-1)
    model = PhysicsInformedHeteroGNN(
        primal_in_dim=primal_in,
        dual_in_dim=dual_in,
        hidden_dim=args.hidden_dim,
        primal_out_dim=primal_in,
        num_layers=args.num_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    edge_p2p = data["primal", "p2p", "primal"].edge_index

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(args.log_dir / "step4_hetero_gnn")

    epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="ep")
    for epoch in epoch_bar:
        model.train()
        opt.zero_grad(set_to_none=True)
        pred_primal = model(data, x_dict)
        loss, data_term, phys_term = physics_informed_total_loss(
            pred_primal,
            target_primal,
            edge_p2p,
            n_primal_vertices=meta.n_primal_vertices,
            lambda_physics=args.lambda_physics,
        )
        loss.backward()
        opt.step()

        writer.add_scalar("loss/total", float(loss.detach()), epoch)
        writer.add_scalar("loss/data_mse", float(data_term), epoch)
        writer.add_scalar("loss/physics", float(phys_term), epoch)

        epoch_bar.set_postfix(
            total=f"{float(loss.detach()):.5f}",
            data=f"{float(data_term):.5f}",
            phys=f"{float(phys_term):.8f}",
        )

    writer.close()

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": meta,
            "train_args": vars(args),
            "primal_in_dim": primal_in,
            "dual_in_dim": dual_in,
        },
        CHECKPOINT,
    )
    tqdm.write(f"Saved model weights to {CHECKPOINT}")


if __name__ == "__main__":
    main()
