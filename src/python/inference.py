"""
学習済み PhysicsHeteroGNN の自己回帰ロールアウトと推論時間計測。
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

PYDIR = Path(__file__).resolve().parent
if str(PYDIR) not in sys.path:
    sys.path.insert(0, str(PYDIR))

from dataset import (
    DUAL,
    PRIMAL,
    E_DUAL,
    E_P2D,
    E_PRIMAL,
    _as_long_edge_index,
    default_interim_path,
    load_interim_json,
)
from model import PhysicsHeteroGNN


def _project_root() -> Path:
    return PYDIR.parent.parent


def edge_dict_from_raw(raw: Dict[str, Any], device: torch.device) -> Dict[Tuple[str, str, str], Tensor]:
    top = raw["topology"]
    d = {
        E_PRIMAL: _as_long_edge_index(top["edge_index"]),
        E_DUAL: _as_long_edge_index(top["dual_edge_index"]),
        E_P2D: _as_long_edge_index(top["primal_to_dual_edge_index"]),
    }
    return {k: v.to(device) for k, v in d.items()}


def load_model_for_inference(
    ckpt_path: Path, device: torch.device
) -> Tuple[PhysicsHeteroGNN, Dict[str, Any]]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    model = PhysicsHeteroGNN(
        in_dim_primal=int(meta["in_dim_primal"]),
        in_dim_dual=int(meta["in_dim_dual"]),
        out_dim=int(meta["out_dim"]),
        hidden=int(meta["hidden"]),
        num_layers=int(meta["num_layers"]),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, meta


@torch.inference_mode()
def autoregressive_rollout(
    model: PhysicsHeteroGNN,
    edge_dict: Dict[Tuple[str, str, str], Tensor],
    *,
    nfs_p: np.ndarray,
    nfs_d: np.ndarray,
    history_len: int,
    t_start: int,
    rollout_steps: int,
    device: torch.device,
    use_profiler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    戻り値: primal_pred [S,N,F], dual_pred [S,Nd,F], total_wall_seconds（学習と同じ履歴スタック）
    """
    t, n, f = nfs_p.shape
    _, nd, fd = nfs_d.shape
    assert fd == f
    if t_start + history_len + rollout_steps > t:
        raise ValueError(
            f"JSON の時系列が不足: T={t}, t_start={t_start}, H={history_len}, S={rollout_steps} → 最終需要 index {t_start + history_len + rollout_steps - 1}"
        )

    buf_p: Deque[Tensor] = deque(
        [torch.as_tensor(nfs_p[t_start + i], dtype=torch.float32, device=device) for i in range(history_len)],
        maxlen=history_len,
    )
    buf_d: Deque[Tensor] = deque(
        [torch.as_tensor(nfs_d[t_start + i], dtype=torch.float32, device=device) for i in range(history_len)],
        maxlen=history_len,
    )

    primal_out: List[Tensor] = []
    dual_out: List[Tensor] = []

    def one_step() -> None:
        x_p = torch.cat(list(buf_p), dim=-1)
        x_d = torch.cat(list(buf_d), dim=-1)
        pred = model({PRIMAL: x_p, DUAL: x_d}, edge_dict)
        pp, pd = pred[PRIMAL], pred[DUAL]
        primal_out.append(pp.clone())
        dual_out.append(pd.clone())
        buf_p.append(pp)
        buf_d.append(pd)

    if use_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            t0 = time.perf_counter()
            for _ in range(rollout_steps):
                one_step()
            t1 = time.perf_counter()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))
    else:
        t0 = time.perf_counter()
        for _ in range(rollout_steps):
            one_step()
        t1 = time.perf_counter()

    total_s = t1 - t0
    p_stack = torch.stack(primal_out, dim=0).cpu().numpy().astype(np.float32)
    d_stack = torch.stack(dual_out, dim=0).cpu().numpy().astype(np.float32)
    return p_stack, d_stack, total_s


def main() -> None:
    ap = argparse.ArgumentParser(description="Autoregressive HeteroGNN rollout")
    ap.add_argument("--checkpoint", type=str, default="", help="hetero_gnn_model.pth")
    ap.add_argument("--json", type=str, default="", help="interim JSON（学習と同じ）")
    ap.add_argument("--rollout-steps", type=int, default=20, dest="rollout_steps")
    ap.add_argument("--t-start", type=int, default=0, dest="t_start", help="ウィンドウ開始時刻インデックス")
    ap.add_argument("--history-len", type=int, default=-1, dest="history_len", help="<=0 で ckpt meta を使用")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--profile", action="store_true", help="torch.profiler で 1 周詳細表示")
    args = ap.parse_args()

    root = _project_root()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / "data" / "interim" / "hetero_gnn_model.pth"
    json_path = Path(args.json) if args.json else default_interim_path(root)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, meta = load_model_for_inference(ckpt_path, device)
    h = int(meta["history_len"]) if args.history_len <= 0 else int(args.history_len)
    if h != int(meta["history_len"]):
        print(f"警告: --history-len={h} と学習時 meta.history_len={meta['history_len']} が異なります。")

    raw = load_interim_json(json_path)
    nfs_p = np.asarray(raw["node_features_time_series"], dtype=np.float32)
    nfs_d = np.asarray(raw["dual_node_features_time_series"], dtype=np.float32)
    t = nfs_p.shape[0]
    max_roll = t - (args.t_start + h)
    if args.rollout_steps > max_roll:
        print(f"警告: rollout_steps を {max_roll} に制限（JSON の T={t}, t_start={args.t_start}, H={h}）")
        args.rollout_steps = max_roll
    if args.rollout_steps < 1:
        raise ValueError("ロールアウト可能なステップがありません。")

    edge_dict = edge_dict_from_raw(raw, device)

    primal_pred, dual_pred, total_s = autoregressive_rollout(
        model,
        edge_dict,
        nfs_p=nfs_p,
        nfs_d=nfs_d,
        history_len=h,
        t_start=args.t_start,
        rollout_steps=args.rollout_steps,
        device=device,
        use_profiler=args.profile,
    )

    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    np.save(interim / "rollout_predictions.npy", primal_pred)
    np.save(interim / "rollout_predictions_dual.npy", dual_pred)
    np.savez_compressed(
        interim / "rollout_meta.npz",
        t_start=np.int32(args.t_start),
        history_len=np.int32(h),
        rollout_steps=np.int32(args.rollout_steps),
        time_axis=np.asarray(raw["time"], dtype=np.float64),
        pred_time_index_start=np.int32(args.t_start + h),
    )

    ms_total = total_s * 1000.0
    ms_per = ms_total / max(args.rollout_steps, 1)
    print(f"ロールアウト完了: {args.rollout_steps} ステップ")
    print(f"総推論時間: {ms_total:.3f} ms（自己回帰ループ全体）")
    print(f"1 ステップ平均: {ms_per:.3f} ms / step")
    print(f"保存: {interim / 'rollout_predictions.npy'}  shape={primal_pred.shape}")
    print(f"保存: {interim / 'rollout_predictions_dual.npy'}  shape={dual_pred.shape}")


if __name__ == "__main__":
    main()
