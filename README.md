# Physics-Informed HeteroGNN Surrogate

[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2?style=flat-square&logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

---

## English Version

### Overview

This repository provides a reproducible surrogate pipeline for **multiphysics CFD**: **Discrete Exterior Calculus (DEC)**–aware **Primal and Dual complexes** are exported from **Julia** and consumed by **heterogeneous graph neural networks (HeteroGNNs)** implemented in **Python** with **PyTorch Geometric** (`HeteroConv`, `SAGEConv`). The goal is to retain mesh-native topological priors—rather than collapsing everything into a homogeneous adjacency—and to amortize costly forward solves with neural rollouts suitable for exploratory design and benchmarking.

Architecturally, the project prioritizes **a strict interoperability contract** (JSON), **index-sane boundaries** between 1-based and 0-based stacks, and **compositionality-friendly** decomposition for future multiphysics extensions.

### Key Features

- **Language boundary (Julia ⇄ Python).** Julia owns **mesh generation** and **physics-motivated discrete dynamics**; Python owns **representation learning, training, autoregressive inference, and plotting**. Stacks communicate exclusively through versioned **`physics_gnn_interim_v2`** JSON—**not** coupled runtimes—so workflows remain deterministic and reproducible across machines.

- **Index safety.** Julia indexing is **1-based**; NumPy / PyTorch use **0-based** semantics. Topology arrays are normalized **exactly once** at export (**`utils_export.jl`**) with coherent COO layout for **`edge_index`**, mitigating the ubiquitous class of silent **Primal–Dual misbindings** when crossing language boundaries.

- **Modularity and compositionality.** The repository splits mesh utilities, exporters, PyG loaders, trainers, rollout engines, and visualization—aligned with **compositionality** motifs from applied category theory and easing replacement of discrete models (e.g., richer physics kernels or categorical DSL backends) **without rewriting the entire surrogate stack.**

- **Heterogeneous message passing.** `PhysicsHeteroGNN` routes messages along **within-Primal**, **within-Dual**, and **`primal_to_dual`** relations, capturing DEC-flavored placements of discrete fields rather than homogeneous lumping alone.

- **Physics-informed surrogate term (engineering placeholder).** For incompressible-style regimes we attach a lightweight penalty encouraging **approximately divergence-free kinematics**, formalized loosely as aligning with \(\nabla \cdot \mathbf{v} = 0\) by **approximating divergence-free constraints via cell-wise flux aggregation**: primal velocities are **`scatter_add`–pooled onto dual cells** and penalized—in **spirit only**, not via a DEC-faithful Hodge / star divergence operator suitable for archival verification literature.

- **Closed-loop realism.** Inference supports **autoregressive rollout** (predictions fed back as inputs), surfacing cumulative drift beyond low one-step supervised error—a primary axis for benchmarking learned CFD surrogates.

### Quick Start (Usage)

Run from the **repository root**.

**1 · Generate interim JSON (Julia ≥ 1.9)** — see **[docs/julia_setup.md](docs/julia_setup.md)** for toolchain details.

```bash
julia --project=src/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=src/julia src/julia/03_simulation.jl
```

Produces **`data/interim/v2_step1_ground_truth_toy.json`** with schema **`physics_gnn_interim_v2`**.

**2 · Train (Python ≥ 3.10)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/python/train.py   # checkpoints → data/interim/hetero_gnn_model.pth
```

Example flags:  
`python src/python/train.py --epochs 50 --history-len 1 --lambda-phys 0.1`  
Alternatively with **[uv](https://docs.astral.sh/uv/)**: `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`

**3 · Autoregressive inference + visualization**

```bash
python src/python/inference.py
python src/python/visualize.py --animate
```

Optional shell helper combining venv activation and `juliaup` PATH: **`source scripts/setup_env.sh`**  
Docker: `docker build -t hetero-surrogate-julia .` then `docker run --rm -v "$PWD":/app -w /app hetero-surrogate-julia`

### License

Released under the **MIT License** — see **[LICENSE](LICENSE)**.

---

## 日本語版 (Japanese Version)

### 概要（Overview）

本リポジトリは、**Discrete Exterior Calculus（離散外微分, DEC）** の位相観に沿った **Primal / Dual 複体** を **Julia** 側で生成・書き出し、**Python** と **PyTorch Geometric（`HeteroConv` 等）** による **異種混合グラフニューラルネットワーク（HeteroGNN）** で学習・推論する、**マルチフィジックス CFD 向けニューラルサロゲート**の再現パイプラインです。メッシュを単一の同種グラフに潰し込まず、場の載る位相を保ったまま **幾何学的深層学習**で前向きシミュレーションのコストを下げることを目的とします。

設計上の核は、**言語間の明確な境界**、**JSON による厳密な中間表現（Contract）**、**1-based / 0-based を跨ぐインデックスの安全保障**、および **合成可能性（Compositionality）** を意識した **疎結合モジュール分割** にあります。

### 主な特徴（Key Features）

- **言語間の境界（Language Boundary）**  
  **Julia**：メッシュ生成・離散物理学に基づくシミュレーション／エクスポートを担当。**Python（PyG）**：表現学習・学習・自己回帰推論・可視化を担当。両者はプロセス内結合ではなく、版付き **`physics_gnn_interim_v2`** **JSON Contract** で連携します。

- **インデックスの安全保障（Index Safety）**  
  Julia は **1-based**、NumPy／PyTorch は **0-based** です。**`utils_export.jl`** において変換と COO の取り決めを **一回きりで集約**し、**Dual と Primal の対応だけがずれるサイレントバグ** を構造的に抑え込みます。

- **可読性とモジュール性／合成可能性（Modularity）**  
  メッシュ・エクスポート・データローダ・学習・ロールアウト・可視化が分離され、応用圏論的文脈で重視される **合成可能性（compositionality）** に沿って差し替え可能です。将来的なマルチフィジックスや別離散モデルへの拡張を前提とした **疎結合設計** です。

- **異種グラフによるメッセージパッシング**  
  **Primal 内／Dual 内／Primal→Dual（`primal_to_dual`）** の複数リレーション上で情報を伝播し、単一同種モデルだけでは捉えにくい位相情報をモデル入力に明示します。

- **Physics-Informed 損失項（現在はプレースホルダ的性格）**  
  非圧縮流れにおける質量保存 \(\nabla \cdot \mathbf{v} = 0\) を厳密に離散化するのではなく、**Primal の流速成分を Dual セルへ `scatter_add` で寄せセル単位のフラックスバランスへ近づける**ことで **発散ゼロ約束を粗く近似する正則化**を入れています（DEC の Hodge／星作用素まで踏み込んだ厳密式ではありません）。

- **自己回帰ロールアウト（Closed-loop）**  
  **Autoregressive rollout** により、1 ステップ誤差が小さくても長ホライズンで誤差が蓄積する現象を評価可能にしています。

### クイックスタート（Quick Start）

リポジトリ**ルート**で実行してください。

**1 · データ生成（Julia ≥ 1.9）** — **`docs/julia_setup.md`** でツールチェーン詳細を参照。

```bash
julia --project=src/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=src/julia src/julia/03_simulation.jl
```

出力：**`data/interim/v2_step1_ground_truth_toy.json`**（スキーマ **`physics_gnn_interim_v2`**）。

**2 · 学習（Python ≥ 3.10）**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/python/train.py   # → data/interim/hetero_gnn_model.pth
```

例：`python src/python/train.py --epochs 50 --history-len 1 --lambda-phys 0.1`  
**[uv](https://docs.astral.sh/uv/)** 利用：`uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`

**3 · 推論・可視化（自己回帰）**

```bash
python src/python/inference.py
python src/python/visualize.py --animate
```

補助：`source scripts/setup_env.sh`／Docker：`docker build -t hetero-surrogate-julia .` → `docker run --rm -v "$PWD":/app -w /app hetero-surrogate-julia`

### ライセンス（License）

**MIT License** で公開しています。**[LICENSE](LICENSE)** を参照してください。
