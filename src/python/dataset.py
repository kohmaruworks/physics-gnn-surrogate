"""
Julia 由来の Phase2 interim JSON を HeteroData へ変換する。
DEC（離散外微分）用のノード/エッジ型名は一貫してここで定義する。

現行 interim v1: primal のみ。v2 では dual / 追加エッジを同じ API で解釈可能。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

# ノード・エッジ型（HeteroData キー）
PRIMAL = "primal_node"
DUAL = "dual_node"
E_PRIMAL = (PRIMAL, "adjacent_to", PRIMAL)  # 三角形メッシュ上の面内近接
E_DUAL = (DUAL, "adjacent_to", DUAL)  # 双対格子の辺
E_P2D = (PRIMAL, "primal_to_dual", DUAL)  # 原始→双対（例: 面に属する角）
E_D2P = (DUAL, "dual_to_primal", PRIMAL)  # 学習時メッセージ戻し用（データ上は p2d の転置）

SCHEMA_V1 = "physics_gnn_phase2_interim_v1"
SCHEMA_V2 = "physics_gnn_phase2_interim_v2"


# --- 厳格な型変換 -------------------------------------------------------------------------


def _as_long_edge_index(edge_index: Union[Sequence[int], List[List[int]], np.ndarray, Tensor]) -> Tensor:
    """COO: [2, E], dtype int64, 各エントリ非負。入力は 1D 列挙 または [2,E] 相当。"""
    if isinstance(edge_index, Tensor):
        t = edge_index
    else:
        arr = np.asarray(edge_index, dtype=np.int64)
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError("edge_index 1D 列挙の長さは偶数である必要があります。")
            t = torch.from_numpy(arr.reshape(2, -1).copy())
        elif arr.ndim == 2 and arr.shape[0] == 2:
            t = torch.from_numpy(arr.copy())
        else:
            raise ValueError(f"edge_index 形状が不正: {arr.shape}")
    t = t.contiguous()
    if t.dtype != torch.long:
        t = t.long()
    if t.dim() != 2 or t.size(0) != 2:
        raise ValueError(f"edge_index は [2, E] である必要がありますが {tuple(t.shape)}")
    if bool((t < 0).any()):
        raise ValueError("edge_index に負のインデックスが含まれます。")
    return t


def _as_float32_features(x: Any) -> Tensor:
    t = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    if t.dim() < 1:
        t = t.view(1, -1)
    return t.contiguous()


# --- スケール正規化（スケルトン） -------------------------------------------------------------


@dataclass
class StandardizeStats:
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    eps: float = 1e-6

    def to_torch(self) -> Tuple[Tensor, Tensor]:
        m = torch.from_numpy(self.mean.astype(np.float32))
        s = torch.from_numpy(self.std.astype(np.float32)).clamp_min(self.eps)
        return m, s


def fit_feature_standardize(
    feat_stack: np.ndarray, feature_names: Optional[Sequence[str]] = None, eps: float = 1e-6
) -> StandardizeStats:
    """
    feat_stack: [num_samples, F] 等、先頭次元で結合した学習用特徴。
    列ごとの平均・分散で標準化。
    """
    m = np.mean(feat_stack, axis=0, dtype=np.float64)
    s = np.std(feat_stack, axis=0, dtype=np.float64)
    if feature_names is None:
        feature_names = []
    return StandardizeStats(mean=m.astype(np.float32), std=s.astype(np.float32), feature_names=list(feature_names), eps=eps)


def apply_feature_standardize(x: Tensor, stats: StandardizeStats) -> Tensor:
    m, s = stats.to_torch()
    while m.dim() < x.dim():
        m, s = m.unsqueeze(0), s.unsqueeze(0)
    return (x - m) / s


# --- 特徴リーク防止 -------------------------------------------------------------------------


@dataclass
class TemporalSplitConfig:
    """入力は最大で target_time_index-1 まで。target は厳密に未来時刻。"""

    target_time_index: int
    history_start: int = 0

    def __post_init__(self) -> None:
        if self.target_time_index <= self.history_start:
            raise ValueError("target_time_index は history_start より大きい必要があります（未来予測）。")


def check_no_future_leakage(
    time_values: np.ndarray, history_indices: List[int], target_index: int, *, time_tol: float = 1e-9
) -> None:
    """入力に使う時刻の最大が、ターゲット時刻**未満**であることを厳密に検査（未来ラベル混入防止）。"""
    if not history_indices:
        raise ValueError("history_indices が空です。")
    t_hist_max = float(np.max(time_values[history_indices]))
    t_tgt = float(time_values[target_index])
    if t_hist_max >= t_tgt - time_tol:
        raise ValueError(
            f"特徴リーク: 入力の最終時刻 (max={t_hist_max}) >= ターゲット時刻 ({t_tgt})。"
        )


def stack_input_features(
    node_features: np.ndarray, history_indices: Sequence[int], order: str = "time_first"
) -> Tensor:
    """
    node_features: [T, N, F]
    返り値: [N, L*F] L=len(history)。concat で時系列積層特徴。
    """
    sl = node_features[history_indices, ...]
    if order == "time_first":
        t, n, f = sl.shape
        return _as_float32_features(sl.reshape(n, t * f))
    raise NotImplementedError(order)


# --- トポロジー -----------------------------------------------------------------------------


def flat_to_triangle_coo(flat: Sequence[int], *, n_nodes: int) -> Optional[Tensor]:
    """0-based 角インデックスのフラット [i0,j0,k0, i1,j1,k1, ...] -> [3, T]。"""
    a = np.asarray(flat, dtype=np.int64)
    if a.size == 0:
        return None
    if a.size % 3 != 0:
        raise ValueError("triangles 配列の長さは 3 の倍数である必要があります。")
    t = torch.from_numpy(a.reshape(3, -1).copy())
    nmax = int(t.max().item()) if t.numel() else -1
    if nmax >= n_nodes or bool((t < 0).any()):
        raise ValueError("triangles インデックスが num_nodes 範囲外です。")
    return t.long().contiguous()


# --- ペイロード解釈 ------------------------------------------------------------------------


def default_interim_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return root / "data" / "interim" / "phase2_step1_ground_truth_toy.json"


def load_interim_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hetero_from_interim(
    raw: Dict[str, Any],
    *,
    config: Optional[TemporalSplitConfig] = None,
) -> HeteroData:
    """
    単一の (入力状態, 予測ターゲット) ペアを 1 グラフに格納。
    config: None の場合は最終時刻前を入力・最後をターゲット（1 ステップ先読みのデモ）。
    """
    schema = str(raw.get("schema", ""))
    num_nodes = int(raw["num_nodes"])
    nfn = list(raw.get("node_feature_names", []))
    ts = np.asarray(raw["time"], dtype=np.float64)
    nfs = np.asarray(raw["node_features_time_series"], dtype=np.float32)
    if nfs.ndim != 3:
        raise ValueError("node_features_time_series は [T, N, F] を想定。")
    t, n, f = nfs.shape
    if nfn and len(nfn) != f:
        raise ValueError("node_feature_names 件数と特徴次元 F が一致しません。")
    if n != num_nodes or t != len(ts):
        raise ValueError("時系列長・節点数が一致しません。")
    fdim = f

    t_final = t - 1
    if config is None:
        if t_final < 1:
            raise ValueError("時刻が 1 点しかなく、1 手先の予測用ペアを作れません。")
        # デフォルト: 時刻 0 ～ t_final-1 を入力、時刻 t_final をターゲット（x に未来は含まない）
        history_idx = list(range(0, t_final))
    else:
        cfg = config
        history_idx = [i for i in range(cfg.history_start, cfg.target_time_index)]
        if not history_idx:
            raise ValueError("history 区間が空です。")
        t_final = cfg.target_time_index

    check_no_future_leakage(ts, history_idx, t_final)
    # 入力: 過去のみ。ターゲット: 厳密に target 時刻
    x_mat = stack_input_features(nfs, history_idx, order="time_first")
    y_vec = _as_float32_features(nfs[t_final, :, :])  # [N, F]
    if x_mat.size(0) != num_nodes or y_vec.size(0) != num_nodes or y_vec.size(1) != fdim or x_mat.size(1) != len(history_idx) * fdim:
        raise ValueError("内部テンソル形状不整合。")

    top = raw.get("topology", {})
    e_p = _as_long_edge_index(top["edge_index"])
    if int(e_p.max().item()) >= num_nodes or int(e_p.min().item()) < 0:
        raise ValueError("primal edge_index が num_nodes 範囲外。")

    data = HeteroData()
    data[PRIMAL].x = x_mat
    data[PRIMAL].y = y_vec
    data[E_PRIMAL].edge_index = e_p
    if schema in ("", SCHEMA_V1):
        _attach_interim_v1_extras(data, top, n_nodes=num_nodes, flat_tri=top.get("triangles", []))
    elif schema == SCHEMA_V2:
        n_dual_exp = int(raw["num_dual_nodes"])
        dual_precomputed = False
        dual_ts = raw.get("dual_node_features_time_series", None)
        if dual_ts is not None and n_dual_exp > 0:
            nfs_d = np.asarray(dual_ts, dtype=np.float32)
            if nfs_d.ndim != 3:
                raise ValueError("dual_node_features_time_series は [T, n_dual, F] を想定。")
            t_d, n_d, f_d = nfs_d.shape
            if t_d != t:
                raise ValueError("dual の時系列長 T が primal と不一致。")
            if n_d != n_dual_exp:
                raise ValueError("num_dual_nodes と dual 時系列の n_dual が不一致。")
            x_d = stack_input_features(nfs_d, history_idx, order="time_first")
            y_d = _as_float32_features(nfs_d[t_final, :, :])
            if x_d.size(0) != n_dual_exp or y_d.size(0) != n_dual_exp or y_d.size(1) != f_d:
                raise ValueError("dual テンソル形状不整合。")
            data[DUAL].x = x_d
            data[DUAL].y = y_d
            dual_precomputed = True
        _attach_from_interim_v2(data, raw, top, num_nodes, skip_dual_xy=dual_precomputed)
    else:
        raise ValueError(f"未対応 schema: {schema}")
    return data


def get_triangle_coo_from_raw(raw: Dict[str, Any], *, n_nodes: int) -> Optional[Tensor]:
    """JSON の topology.triangles から [3, F] 長期テンソル（0-based）。GNN 外の可視化・参照用。"""
    top = raw.get("topology", {})
    return flat_to_triangle_coo(top.get("triangles", []), n_nodes=n_nodes)


def _attach_interim_v1_extras(
    data: HeteroData, top: Dict[str, Any], *, n_nodes: int, flat_tri: Sequence[int]
) -> None:
    """v1: 双対は JSON 未拡張のためここでは何も足さない。三角形は get_triangle_coo_from_raw 参照。top/tri 引数は v1→v2 移行用フック。"""
    assert n_nodes == data[PRIMAL].x.size(0)
    assert "edge_index" in top
    _ = flat_to_triangle_coo(flat_tri, n_nodes=n_nodes)  # 形状検査のみ（GNN 入力には出さない）


def _attach_from_interim_v2(
    data: HeteroData,
    raw: Dict[str, Any],
    top: Dict[str, Any],
    n_primal: int,
    *,
    skip_dual_xy: bool = False,
) -> None:
    """v2: dual の x/y は build_hetero 側で時系列から積む場合あり（skip_dual_xy=True）。"""
    n_dual = int(raw["num_dual_nodes"])
    if n_dual == 0:
        data[DUAL].num_nodes = 0
    else:
        if not skip_dual_xy:
            dfeat = raw.get("dual_node_features", None)
            if dfeat is None:
                raise ValueError("v2: num_dual_nodes>0 で dual 時系列が無い場合は dual_node_features が必要です。")
            ndsf = _as_float32_features(dfeat)
            if ndsf.dim() == 1:
                ndsf = ndsf.view(-1, 1)
            if int(ndsf.size(0)) != n_dual:
                raise ValueError("dual 特徴量行数と num_dual_nodes が一致しません。")
            data[DUAL].x = ndsf
            data[DUAL].y = ndsf.clone()

    if "dual_edge_index" in top and top["dual_edge_index"] is not None:
        d_ei = _as_long_edge_index(top["dual_edge_index"])
        if n_dual and (int(d_ei.max().item()) >= n_dual or int(d_ei.min().item()) < 0):
            raise ValueError("dual_edge_index が num_dual 範囲外。")
        if not n_dual and d_ei.numel() > 0:
            raise ValueError("n_dual=0 だが dual_edge_index が非空。")
        data[E_DUAL].edge_index = d_ei

    if "primal_to_dual_edge_index" in top and top["primal_to_dual_edge_index"] is not None:
        p2d = _as_long_edge_index(top["primal_to_dual_edge_index"])
        if p2d.numel() > 0:
            if (p2d[0] >= n_primal).any() or (p2d[1] >= n_dual).any() or (p2d < 0).any():
                raise ValueError("primal_to_dual インデックス範囲外。")
        data[E_P2D].edge_index = p2d

    if "edge_index" in top:
        p_ei = _as_long_edge_index(top["edge_index"])
        if int(p_ei.max().item()) >= n_primal:
            raise ValueError("primal edge_index 範囲外 (v2)。")
        data[E_PRIMAL].edge_index = p_ei


# --- 複数 HeteroData サンプル（スライド窓） ---------------------------------------------------


def list_hetero_from_interim(
    path: Optional[Path] = None,
    raw: Optional[Dict[str, Any]] = None,
    *,
    history_len: int = 1,
    step: int = 1,
) -> List[HeteroData]:
    """
    1 手先予測の教師ありサンプル列。

    各サンプル k: 入力=時刻 t..t+history_len-1, ターゲット= t+history_len
    リーク防止: 常に target_time_index より前のフレームのみを入力に使用。
    """
    p = path or default_interim_path()
    d = raw if raw is not None else load_interim_json(p)
    ts = np.asarray(d["time"], dtype=np.float64)
    nfs = np.asarray(d["node_features_time_series"], dtype=np.float32)
    t_max = len(ts) - 1
    out: List[HeteroData] = []
    tgt = history_len
    while tgt <= t_max:
        h0 = tgt - history_len
        if h0 < 0:
            break
        cfg = TemporalSplitConfig(target_time_index=tgt, history_start=h0)
        out.append(build_hetero_from_interim(d, config=cfg))
        tgt += step
    if not out:
        raise ValueError("生成されたサンプルが 0 件。history_len またはデータ長を見直してください。")
    return out


def make_hetero_dataloader(
    data_list: Sequence[HeteroData],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> PyGDataLoader:
    return PyGDataLoader(
        list(data_list),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
