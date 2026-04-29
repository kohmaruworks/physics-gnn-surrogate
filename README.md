# Physics-Informed HeteroGNN Surrogate (Phase 2)

Julia と PyTorch Geometric（PyG）を連携させ、離散外微分（DEC）のトポロジーを学習するマルチフィジックス CFD 向けの AI サロゲートモデルです。

## 🚀 プロジェクト概要

本プロジェクトは、物理シミュレーションの計算コストを劇的に削減する AI サロゲート（代替モデル）の構築を目指しています。Phase 2 では、同種グラフ（Homogeneous Graph）ではなく、DEC に基づく **Primal / Dual 構造を持った異種混合グラフ（Heterogeneous Graph）** を採用し、幾何学的深層学習によって物理法則の位相的特徴を捉えるパイプラインを構築しました。

### ✨ 主な特徴

* **言語間の完全分離:** 物理モデリングとメッシュ生成（真実の源泉）は Julia エコシステムで構築し、AI の表現学習は Python（PyG）で行う疎結合アーキテクチャです。
* **JSON Contract V2:** Julia と Python 間で Primal / Dual / Primal-to-Dual の関係性を安全に受け渡すため、0-based インデックス補正を組み込んだ厳格な中間表現スキーマを実装しています。
* **双方向メッセージパッシング:** `HeteroConv` を用い、データ容量を節約しつつモデル内で動的に逆辺を生成し、空間全体の情報を効率的に伝播します。
* **自己回帰ロールアウト評価:** 1 ステップの予測にとどまらず、予測値を次ステップの入力とする自己回帰（Autoregressive）推論を実装し、長期予測時の誤差蓄積を定量的に評価できます。

## 📂 ディレクトリ構成

```text
physics-gnn-surrogate-phase2/
├── data/
│   └── interim/        # Julia から出力される JSON、学習済み .pth、推論 .npy / .npz、比較図・GIF
├── scripts/
│   └── phase2_env.sh   # venv + Julia PATH 用ヘルパ（任意）
├── src/
│   ├── julia/          # [Data Generation] メッシュ生成、物理モデリング、シミュレーション
│   └── python/         # [AI Surrogate] PyG によるデータロード、学習、推論、可視化
├── Dockerfile          # （任意）Julia をホストに入れない場合
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ 環境構築と実行方法

※ 以下はリポジトリの**ルート**をカレントにした前提です。

### 1. データ生成（Julia）

Julia 1.9 以上が必要です。`CombinatorialSpaces` 等を用いてグラウンドトゥルースを生成します。

```bash
julia --project=src/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=src/julia src/julia/03_simulation.jl
```

> `data/interim/phase2_step1_ground_truth_toy.json` が生成されます（スキーマは **V2**）。

### 2. モデル学習（Python）

Python 3.10 以上を推奨します。仮想環境を作成し、依存関係を入れてから学習を実行します。

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/python/train.py
```

> 学習済みモデルは `data/interim/hetero_gnn_model.pth` に保存されます。  
> オプション例: `python src/python/train.py --epochs 50 --history-len 1 --lambda-phys 0.1`（`src/python` に移動して `python train.py ...` でも同じです）。

依存のインストールに [`uv`](https://docs.astral.sh/uv/) を使う場合は `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt` でも構いません。

### 3. 推論と可視化（Python）

学習したモデルで自己回帰推論を行い、結果を可視化・GIF 化します。

```bash
python src/python/inference.py
python src/python/visualize.py --animate
```

> `data/interim/rollout_predictions.npy` 等の推論結果および `data/interim/comparison_plot.png`、`--animate` 時は `data/interim/rollout_animation.gif` が生成されます。推論スクリプトは標準出力に推論時間（ms）の目安も表示します。

## 📊 実行結果ハイライト

* **推論速度の大幅な向上:** 数値ソルバーと比較して、1 ステップあたりの計算時間をミリ秒オーダー（ms/step）まで短縮できます。
* **誤差の観察:** 学習直後のステップでは MSE が非常に低く抑えられていますが、10 ステップ以上の自己回帰ロールアウトにおいて、ベクトル場（流速）の微小な発散やスカラー場（温度）の数値拡散の傾向が確認されました。これは自己回帰 GNN 特有の課題であり、次フェーズ（Phase 3）でのアーキテクチャ改良の基盤となります。

## 🔗 次のステップ（Phase 3 展望）

* 複雑な CAD 由来の非構造メッシュへの対応とゼロショット汎化性能の検証。
* 離散外微分の演算子（外微分 *d*、余微分 *δ*）を厳密に組み込んだ高度な Physics-Informed Loss の実装。

## ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。
