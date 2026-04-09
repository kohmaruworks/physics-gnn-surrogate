---
title: "サロゲートモデル構築(1)：単純なばね-質量系における検証と、Julia-Python間のデータブリッジ設計"
type: "tech"
topics: ["Julia", "Python", "PyTorch Geometric", "Catlab", "サロゲートモデル"]
published: false
---

# サロゲートモデル構築(1)：単純なばね-質量系における検証と、Julia-Python間のデータブリッジ設計

## 1. はじめに

数値シミュレーションは設計・制御・最適化の中核であるが、高忠実度モデルは計算コストが大きく、パラメータスイープやオンライン推論には向かない場面がある。そこで**サロゲートモデル**（代理モデル）として、ニューラルネットや GNN が入出力関係を近似し、推論を高速化するアプローチが広く検討されている。

本連載の目的は、単に「物理っぽいデータで学習する」ことにとどまらず、**応用圏論（Catlab.jl 等）でスキーマを固定した構造**と、**PyTorch Geometric（PyG）上の学習パイプライン**を接続し、**再現可能で拡張しやすい Physics-Informed AI アーキテクチャ**を組み立てることにある。第1回では、最も単純なばね–質量系を題材に学習比較の落とし穴を述べたうえで、Julia と Python の間を JSON で橋渡しする設計を詳述する。

## 2. 検証1：簡易的なばね-質量系での学習比較

### 2.1 実施内容（想定）

質点をノード、バネ結合を辺とするグラフ上で、各ノードに位置・速度などの特徴ベクトルを載せる。教師信号は、フックの法則に基づく力と時間積分（例：オイラー法や高次ソルバ）から得られる**次ステップの状態**などとする。

ここで**ノード数 $N$ を固定**（例：$N=100$ のチェーン）し、同一トポロジ上で乱数初期条件から $(x, y)$ のペアを大量に生成する。比較対象は次の二つである。

- **構造を持たない MLP**：全ノード特徴を一列に連結し、$\mathbb{R}^{N d} \to \mathbb{R}^{N d}$ の写像として学習する。
- **GNN（例：GCN）**：`edge_index` に沿って局所メッセージを送る。

```python
# プレースホルダ：固定 N の MLP（入力次元 N * feat_dim が固定）
# model_mlp = NaiveMLP(num_nodes=N_fixed, feat_dim=d, hidden_dim=H)
# loss = mse(model_mlp(x_flat), y_flat)
```

```python
# プレースホルダ：同一グラフ上の GNN
# model_gnn = CategoryInformedGNN(in_channels=d, hidden=H, out_channels=d)
# loss = mse(model_gnn(x, edge_index), y)
```

### 2.2 結果と考察：固定環境における「AIの罠」

固定された $N$ と固定された隣接構造のもとでは、教師写像が実質**有限次元ベクトル空間上の関数**に帰着する。MLP は（十分な幅と深さがあれば）その写像を**パラメータに吸収**しやすく、隣接関係を明示的に入力に含めなくても、データ分布の中で**事実上「グラフを暗記」した多層パーセプトロン**として振る舞いうる。

一方、GNN は重み共有と局所集約により帰納バイアスが入るため、タスクによっては MLP より学習が難しい・損失が残ることもある。重要なのは、**固定 $N$ のベンチマークだけでは「構造を利用しているから優れた」ことを証明できない**という事実である。本連載はこの誠実な前提から出発し、第2回以降でスケール変動やマルチフィジックスへ進む。

## 3. システム統合アーキテクチャ：Catlab.jl と PyTorch Geometric のデータブリッジ

Catlab の `Catlab.Graphs.Graph` は**有向マルチグラフ**としてグラフを表す。PyG の `torch_geometric.data.Data` は稀疏な `edge_index`（形状 $[2, |\mathcal{E}|]$）とノード特徴 `x` で表す。両者の間には、少なくとも次の差がある。

| 観点 | Catlab（Julia） | PyTorch Geometric（Python） |
|------|-----------------|------------------------------|
| 頂点 ID | **1-based**（Julia 慣習） | **0-based**（`edge_index` の慣習） |
| 辺 | 辺 ID ごとに `src`, `tgt` | 各列が $(\mathrm{source}, \mathrm{target})$ |
| 孤立頂点 | `nv(g)` で明示 | `num_nodes` の明示が望ましい |

本リポジトリでは **JSON を中間表現**とし、Julia 側でグラフをシリアライズし、Python 側で `Data` に復元する。変換の要点は、**JSON 上の頂点番号を最初から 0-based に固定する**ことで、Python 側で `edge_index -= 1` のような二重補正を避けることにある。

### 3.1 JSON スキーマ（`catlab_directed_graph_v1`）

| キー | 型 | 必須 | 説明 |
|------|-----|------|------|
| `format` | 文字列 | はい | `"catlab_directed_graph_v1"` |
| `num_nodes` | 整数 | はい | 頂点数（個数なので 0/1-based は無関係） |
| `edges` | 配列 | いいえ | `[[src, tgt], ...]`。**頂点番号は 0-based** |
| `x` | 配列 | いいえ | ノード特徴。行 $i$ が頂点 $i$（0-based） |
| `y` | 配列 | いいえ | 教師（任意） |

### 3.2 Julia 側：`export_catlab_graph_json.jl`

`edges(g)` で辺を列挙し、各辺について Catlab の `src(g, e)`, `tgt(g, e)` を得る。これらは **1-based の頂点 ID** である。JSON では

$$
\texttt{edges} \leftarrow \bigl[ [\texttt{src}-1,\ \texttt{tgt}-1],\ \ldots \bigr]
$$

とし、**一度だけ** 0-based に落とす。`num_nodes` は `nv(g)` をそのまま格納する。`x` が行列のときは、Julia の行 $i$（1-based）が頂点 $i$ に対応する前提でリスト化し、Python では行 $i-1$ が頂点 $i-1$ と対応する。

```julia
# プレースホルダ：Julia での書き出し（実装は export_catlab_graph_json.jl を参照）
# save_catlab_graph_json("graph.json", g; x=x_matrix, y=y_matrix)
```

### 3.3 Python 側：`import_catlab_json_to_pyg.py`

1. `format` を検証する。  
2. `edges` を `torch.tensor` で読み、**`.t().contiguous()`** して形状 $[2, |\mathcal{E}|]$ の `edge_index` にする（COO：1 行目が source、2 行目が target）。  
3. `Data(edge_index=..., num_nodes=...)` を構築し、`x` / `y` があれば `float32` テンソルにする。

```python
# プレースホルダ
# from import_catlab_json_to_pyg import catlab_json_to_data
# data = catlab_json_to_data("graph_from_catlab.json")
```

### 3.4 なぜ中間形式を 0-based に固定するか

- **受け手が PyG**であるため、仕様を PyG のセマンティクスに合わせると下流が単純になる。  
- **オフセット変換は Julia の書き出し時に一度だけ**行えば、JSON を読む他ツールにも「頂点 ID は 0-based」と宣言しやすい。  
- `x` の行順と `edges` のインデックスを同じ基準に置けるため、**孤立頂点を含むグラフ**でも解釈が一貫する。

### 3.5 注意点

頂点削除などで **ID が非連続**になるグラフをそのまま書き出すと、0-based 化後も **$0..N-1$ に圧縮されていない**ラベルになりうる。PyG は通常、頂点インデックスが $0,\ldots,N-1$ に揃っていることを期待するため、そのような場合は**書き出し前に頂点を正規化する**か、Python 側で**リマッピング**する別パイプラインが必要である。本ブリッジは **連続 ID $1:n$ の典型的なグラフ**を想定している。

逆方向（PyG → Catlab）では、読み込み時に辺の端点へ $+1$ して `add_edge!(g, s+1, t+1)` のように戻す必要がある。

### 3.6 実行例

```bash
# プレースホルダ：Julia 依存の解決
# julia --project=. -e 'using Pkg; Pkg.instantiate()'

# JSON 生成（スクリプト例）
# julia --project=. export_catlab_graph_json.jl

# Python で Data へ
# python3 import_catlab_json_to_pyg.py graph_from_catlab.json
```

---

第2回では、ノード数が変わる現場要件に対し、**固定次元 MLP が破綻し、GNN が推論を継続できる**ことをスケール変動テストで見る。第3回では、バネとダンパーが混在するマルチフィジックスと、**HeteroData / HeteroConv による辺タイプ分離**へ進む。
