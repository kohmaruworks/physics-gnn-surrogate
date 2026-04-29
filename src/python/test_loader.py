"""
dataset.py の動作検証: HeteroData の validate、テンソル形状、DataLoader の反復。
"""
from __future__ import annotations

import sys
from pathlib import Path

PYDIR = Path(__file__).resolve().parent
if str(PYDIR) not in sys.path:
    sys.path.insert(0, str(PYDIR))

import torch

from dataset import (
    E_DUAL,
    E_P2D,
    E_PRIMAL,
    PRIMAL,
    DUAL,
    build_hetero_from_interim,
    default_interim_path,
    get_triangle_coo_from_raw,
    list_hetero_from_interim,
    load_interim_json,
    make_hetero_dataloader,
)


def _check_tensor(name: str, t: torch.Tensor, *, exp_float: bool = True) -> None:
    print(f"  [{name}] shape={tuple(t.shape)} dtype={t.dtype}")
    if exp_float and t.dtype != torch.float32:
        raise AssertionError(f"{name}: float32 期待, got {t.dtype}")
    if not exp_float and t.dtype != torch.long:
        raise AssertionError(f"{name}: long 期待, got {t.dtype}")


def main() -> None:
    path = default_interim_path()
    print(f"JSON: {path}")
    raw = load_interim_json(path)
    schema = raw.get("schema", "")
    print(f"schema: {schema}")

    tri = get_triangle_coo_from_raw(raw, n_nodes=int(raw["num_nodes"]))
    if tri is not None:
        print(f"triangles COO (参考): shape={tuple(tri.shape)} dtype={tri.dtype}")

    # 単一グラフ（全過去→最終時刻）
    h0 = build_hetero_from_interim(raw, config=None)
    print("\n--- build_hetero_from_interim(config=None) ---")
    ok = h0.validate(raise_on_error=False)
    print(f"HeteroData.validate(raise_on_error=False) -> {ok}")
    if not ok:
        # 詳細は PyG が ValueError を投げるため raise_on_error=True で再実行
        h0.validate(raise_on_error=True)
    print(f"node_types: {h0.node_types}")
    print(f"edge_types: {h0.edge_types}")
    _check_tensor(f"{PRIMAL}.x", h0[PRIMAL].x)
    _check_tensor(f"{PRIMAL}.y", h0[PRIMAL].y)
    _check_tensor(f"{E_PRIMAL}.edge_index", h0[E_PRIMAL].edge_index, exp_float=False)

    if schema == "physics_gnn_phase2_interim_v2" and DUAL in h0.node_types:
        _check_tensor(f"{DUAL}.x", h0[DUAL].x)
        _check_tensor(f"{DUAL}.y", h0[DUAL].y)
        _check_tensor(f"{E_DUAL}.edge_index", h0[E_DUAL].edge_index, exp_float=False)
        _check_tensor(f"{E_P2D}.edge_index", h0[E_P2D].edge_index, exp_float=False)
        nd = int(raw["num_dual_nodes"])
        if int(h0[E_DUAL].edge_index.max().item()) >= nd:
            raise AssertionError("dual_edge_index が num_dual 範囲外")

    n = int(raw["num_nodes"])
    ei = h0[E_PRIMAL].edge_index
    if int(ei.max().item()) >= n or int(ei.min().item()) < 0:
        raise AssertionError("edge_index が [0, num_nodes) 外")

    # スライド窓サンプル + DataLoader
    samples = list_hetero_from_interim(raw=raw, history_len=1, step=1)
    print(f"\n--- list_hetero_from_interim(history_len=1) -> {len(samples)} サンプル ---")
    for i, s in enumerate(samples[:3]):
        s.validate(raise_on_error=True)
        print(f"  sample {i}: x {tuple(s[PRIMAL].x.shape)} y {tuple(s[PRIMAL].y.shape)}")

    loader = make_hetero_dataloader(samples, batch_size=2, shuffle=False)
    print(f"\nDataLoader: batch_size=2, batches={len(loader)}")
    for bi, batch in enumerate(loader):
        batch.validate(raise_on_error=True)
        print(f"  batch {bi}: batched {PRIMAL}.x={tuple(batch[PRIMAL].x.shape)}")
        if bi >= 1:
            break

    # v1 のみ dual 未登録
    if DUAL not in h0.node_types:
        print(f"\n注: interim v1 のため {DUAL} / {E_DUAL} / {E_P2D} は未設定。")
    elif schema == "physics_gnn_phase2_interim_v2":
        print(f"\n注: Contract V2 — {DUAL}・{E_DUAL}・{E_P2D} を検証済み。")

    print("\nすべての検査に成功しました。")


if __name__ == "__main__":
    main()
