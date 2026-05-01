"""
HeteroData ローダ + PhysicsHeteroGNN の学習（データ MSE + 発散ペナルティのプレースホルダ）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

PYDIR = Path(__file__).resolve().parent
if str(PYDIR) not in sys.path:
    sys.path.insert(0, str(PYDIR))

from dataset import (
    DUAL,
    PRIMAL,
    E_P2D,
    SCHEMA_V2,
    default_interim_path,
    list_hetero_from_interim,
    load_interim_json,
    make_hetero_dataloader,
)
from model import PhysicsHeteroGNN


def _project_root() -> Path:
    return PYDIR.parent.parent


def divergence_proxy_loss(
    pred_primal: Tensor,
    edge_index_p2d: Tensor,
    num_dual: int,
) -> Tensor:
    r"""
    プレースホルダ: p2d で primal 速度 (vx, vy) を dual セルへ scatter 加算し、
    セルごとの総和が 0 に近いことを要求（非圧縮 \nabla\cdot v=0 の超簡易代理）。
    pred_primal: [N_primal, F], vx=0, vy=1
    edge_index_p2d: [2, E], row0=primal, row1=dual
    """
    if edge_index_p2d.numel() == 0 or num_dual == 0:
        return pred_primal.new_zeros(())
    src = edge_index_p2d[0]
    dst = edge_index_p2d[1]
    flux = pred_primal[src, 0] + pred_primal[src, 1]
    agg = torch.zeros(num_dual, device=pred_primal.device, dtype=pred_primal.dtype)
    agg.scatter_add_(0, dst, flux)
    return agg.pow(2).mean()


def train_epoch(
    model: PhysicsHeteroGNN,
    loader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    lambda_phys: float,
) -> tuple[float, float, float]:
    model.train()
    tot_data = 0.0
    tot_phys = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        x_dict = {PRIMAL: batch[PRIMAL].x, DUAL: batch[DUAL].x}
        edge_dict = {k: batch[k].edge_index for k in batch.edge_types}

        y_primal = batch[PRIMAL].y
        y_dual = batch[DUAL].y

        pred = model(x_dict, edge_dict)
        loss_data = F.mse_loss(pred[PRIMAL], y_primal)
        if y_dual.numel() > 0:
            loss_data = loss_data + F.mse_loss(pred[DUAL], y_dual)

        n_dual = pred[DUAL].size(0)
        loss_phys = divergence_proxy_loss(pred[PRIMAL], batch[E_P2D].edge_index, n_dual)

        loss = loss_data + lambda_phys * loss_phys
        loss.backward()
        opt.step()
        tot_data += float(loss_data.detach())
        tot_phys += float(loss_phys.detach())
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return tot_data / n, tot_phys / n, n


def main() -> None:
    p = argparse.ArgumentParser(description="Physics-informed HeteroGNN (HeteroGNN Edition)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lambda-phys", type=float, default=0.1, dest="lambda_phys")
    p.add_argument("--history-len", type=int, default=1, dest="history_len")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument("--json", type=str, default="", help="interim JSON（空ならデフォルト）")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    json_path = Path(args.json) if args.json else default_interim_path()
    raw = load_interim_json(json_path)
    if str(raw.get("schema", "")) != SCHEMA_V2:
        print("警告: schema が v2 ではありません。dual / p2d が無いと学習が制限されます。")

    data_list = list_hetero_from_interim(raw=raw, history_len=args.history_len, step=1)
    sample = data_list[0]
    in_p = int(sample[PRIMAL].x.size(-1))
    in_d = int(sample[DUAL].x.size(-1))
    out_dim = int(sample[PRIMAL].y.size(-1))
    assert out_dim == sample[DUAL].y.size(-1), "primal / dual の y 次元が一致していること"

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    loader = make_hetero_dataloader(data_list, batch_size=args.batch_size, shuffle=True)

    model = PhysicsHeteroGNN(
        in_dim_primal=in_p,
        in_dim_dual=in_d,
        out_dim=out_dim,
        hidden=args.hidden,
        num_layers=args.layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        d_loss, ph_loss, nb = train_epoch(model, loader, opt, device, args.lambda_phys)
        total = d_loss + args.lambda_phys * ph_loss
        print(
            f"epoch {ep:03d} | L_data={d_loss:.6f} | L_phys={ph_loss:.6f} | "
            f"L_total={total:.6f} (λ={args.lambda_phys}) | batches={int(nb)}"
        )

    out_path = _project_root() / "data" / "interim" / "hetero_gnn_model.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "in_dim_primal": in_p,
                "in_dim_dual": in_d,
                "out_dim": out_dim,
                "hidden": args.hidden,
                "num_layers": args.layers,
                "history_len": args.history_len,
                "schema": raw.get("schema", ""),
            },
        },
        out_path,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
