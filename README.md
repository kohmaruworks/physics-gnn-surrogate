# Physics-Informed Surrogate Modeling via Applied Category Theory (ACT-GNN)

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

## Overview

This repository provides a proof-of-concept pipeline demonstrating how **Applied Category Theory (ACT)** and **Graph Neural Networks (GNNs)** can solve critical challenges in physics-informed surrogate modeling. 

Through a series of demonstrations, we prove why standard structure-less AI (MLPs) fail in real-world R&D (scale variance), and why standard Homogeneous GNNs fail in complex environments (multiphysics). We resolve these by introducing a strict categorical schema using **Catlab.jl (Julia)** and bridging it to **PyTorch Geometric (Python)** via a custom JSON pipeline.

> **Related Articles (Japanese)**: 
> This repository is tied to a 3-part article series on Zenn detailing the mathematical proofs and architectural designs.

## Core Demonstrations

### 1. Scale Generalization Test (MLP Crash vs GNN)
**Script**: `demo1_scale_generalization.py`

In real-world R&D, mesh sizes and node counts ($N$) constantly fluctuate. This demo trains models on a small system ($N=10$) and tests them on a larger system ($N=50$).
- **Naive MLP**: Crashes due to matrix size mismatch (`RuntimeError`). Proof that structure-less AI cannot generalize to different scales.
- **GNN**: Successfully performs zero-shot generalization by relying on local message passing regardless of graph size.

### 2. Multiphysics & Category Theory (Homogeneous vs Hetero GNN)
**Scripts**: `demo2_category_multiphysics.py`, `compare_loss_visualization.py`

Simulating a multiphysics environment where multiple forces (e.g., position-dependent **Springs** and velocity-dependent **Dampers**) coexist.
- **Homogeneous GNN**: Fails to distinguish between different physical laws, causing the loss to plateau.
- **Categorical Hetero GNN**: By defining a strict schema in `Catlab.jl` where masses are Objects and springs/dampers are distinct Morphisms, the `HeteroConv` architecture completely isolates the physics channels, achieving rapid and precise convergence.

## Architecture & Data Flow

1. **Categorical Modeling (Julia)**: `export_catlab_graph_json.jl` defines the physical system as a directed multigraph using `Catlab.jl`. 
2. **Data Bridge**: Exports topology and features to a JSON format. We employ a strict **1-based to 0-based index conversion** during export to perfectly align Julia's conventions with Python's.
3. **Deep Learning (Python)**: `import_catlab_json_to_pyg.py` parses the JSON into `torch_geometric.data.HeteroData` objects for training.

## How to Run

### Setup Environment
Ensure you have Julia (with `Catlab.jl`) and Python (with `PyTorch` and `PyTorch Geometric`) installed.

```bash
# Python dependencies
pip install torch torch-geometric matplotlib
```

### Run Demo 1: Scale Generalization
```bash
python demo1_scale_generalization.py
```
*(Watch the MLP throw a dimension mismatch error while the GNN succeeds.)*

### Run Demo 2: Multiphysics Category Theory
```bash
python demo2_category_multiphysics.py
```
This will train both models and output `hetero_loss_comparison.png`, visualizing the overwhelming superiority of the Hetero GNN approach.

---

## 日本語版 (Japanese Version)

### 概要
本リポジトリは、物理シミュレーションをAIで高速化する「サロゲートモデル」の構築において、**応用圏論（Applied Category Theory）**と**グラフニューラルネットワーク（GNN）**の有用性を実証するPoC（概念実証）です。

Zennでの全3回の連載記事と連動しており、構造を持たないAI（MLP）の限界から、マルチフィジックス環境下における圏論的アーキテクチャの圧倒的優位性までをコードで証明します。

### 主要な検証デモ（スクリプト）

#### 1. スケール変動テスト（MLPのクラッシュ証明）
**実行ファイル**: `demo1_scale_generalization.py`
実務ではメッシュの切り直し等でノード数（$N$）が変動します。学習時（$N=10$）と異なる未知のスケール（$N=50$）を入力した際の挙動を比較します。
- **MLP**: 行列積の次元不整合（`RuntimeError`）によりクラッシュします。
- **GNN**: 局所的なメッセージパッシングにより、スケールフリーに推論を完了します。

#### 2. マルチフィジックスと圏論の優位性（バネとダンパーの混在）
**実行ファイル**: `demo2_category_multiphysics.py`, `compare_loss_visualization.py`
バネ（位置に依存）とダンパー（速度に依存）という異なる物理法則が混在する系での学習を比較します。
- **同質GNN**: エッジの種類を区別できず、情報を混同してLossがプラトーに達します。
- **圏論的Hetero GNN**: `Catlab.jl`で質量（Object）とバネ・ダンパー（異なるMorphism）を型レベルで分離。PyGの`HeteroData`として処理することで、物理法則ごとの伝達経路が分離され、圧倒的な収束を見せます。

### 実行方法

**デモ1：スケール変動テストの実行**
```bash
python demo1_scale_generalization.py
```

**デモ2：マルチフィジックス圏論比較の実行**
```bash
python demo2_category_multiphysics.py
```
実行後、同質GNNとHetero GNNの学習曲線を比較したグラフ画像（`hetero_loss_comparison.png`）が出力されます。

### データブリッジ設計について
Julia（1-based）とPython（0-based）のインデックスのズレを吸収するため、`export_catlab_graph_json.jl` にてJSON出力時に0-basedへの変換を一度だけ行い、Python側の `import_catlab_json_to_pyg.py` でクリーンにPyGのDataオブジェクトへと復元するアーキテクチャを採用しています。
