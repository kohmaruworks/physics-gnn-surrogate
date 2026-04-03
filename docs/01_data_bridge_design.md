# Catlab.jl 有向グラフと PyTorch Geometric の JSON ブリッジ

このドキュメントは、`export_catlab_graph_json.jl`（Julia）と `import_catlab_json_to_pyg.py`（Python）の役割、データ形式、および **Julia の 1-based 頂点番号を JSON 上で 0-based に揃える設計意図**を説明する。

---

## 1. 全体像

Catlab の `Catlab.Graphs.Graph` は **有向マルチグラフ**（辺に ID があり、同一頂点対に複数辺を持てる）として表現される。PyTorch Geometric（PyG）の `torch_geometric.data.Data` は、稀疏な `edge_index`（形状 `[2, num_edges]`）と任意のノード特徴 `x` でグラフを表す。

両者の間には次の差がある。

| 観点 | Catlab（Julia） | PyTorch Geometric（Python） |
|------|-----------------|------------------------------|
| 頂点 ID | **1-based**（Julia 慣習・Catlab の `V` の添字） | **0-based**（`edge_index` の慣習） |
| 辺の表現 | 辺 ID ごとに `src(g,e)`, `tgt(g,e)` | 各列が `(source, target)` の整数ペア |
| 孤立頂点 | `nv(g)` で明示 | `num_nodes` の明示が望ましい |

本ブリッジは **JSON を中間表現**とし、Julia 側でグラフ構造をシリアライズし、Python 側で `Data` に復元する。変換の要点は **JSON 上の頂点番号を最初から 0-based に固定する**ことで、Python 側で再変換せずに PyG の期待と一致させることにある。

---

## 2. JSON スキーマ（`catlab_directed_graph_v1`）

| キー | 型 | 必須 | 説明 |
|------|-----|------|------|
| `format` | 文字列 | はい | 識別子。`"catlab_directed_graph_v1"` のみ対応。 |
| `num_nodes` | 整数 | はい | 頂点数。Julia の `nv(g)` に等しい。孤立頂点を含む。 |
| `edges` | 配列の配列 | いいえ | `[[src, tgt], ...]`。**頂点番号は 0-based**。辺が無いときは `[]` または省略可（Python 側で空配列として扱う）。 |
| `x` | 配列の配列 | いいえ | ノード特徴。外側が頂点、内側が特徴次元。行 `i` が頂点 `i`（0-based）に対応。 |

マルチエッジは、`edges` に同じ `[s,t]` が複数行として現れうる。Catlab の辺 ID の順序は `edges(g)` の列挙順に従う。

---

## 3. Julia 側: `export_catlab_graph_json.jl`

### 3.1 依存関係

- `Catlab` / `Catlab.Graphs`: グラフ型と `Graph()`, `add_vertices!`, `add_edges!`, `nv`, `edges`, `src`, `tgt` など。
- `JSON3`: JSON の書き出し（コンパクト出力と `pretty` 整形）。

プロジェクト環境はリポジトリ直下の `Project.toml` で管理し、初回は次で依存を解決する。

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 3.2 `catlab_graph_edges_0based(g)`

`edges(g)` で辺 ID を列挙し、各辺について `src(g, e)` と `tgt(g, e)` を取得する。これらは **Julia / Catlab 上では 1-based の頂点 ID** である。

ここで **両方に `-1` して** `[src-1, tgt-1]` のベクタを `edges` 配列に積む。これが JSON 上の **0-based** エッジリストになる。

### 3.3 `catlab_graph_to_dict(g; x=nothing)`

- `num_nodes` に `nv(g)` をそのまま格納する。値は「個数」であり添字ではないため、1-based / 0-based のオフセットは不要。
- `edges` に上記 `catlab_graph_edges_0based(g)` を格納。
- `x` が与えられた場合:
  - `AbstractMatrix` なら、行 `i`（Julia の 1-based 行）が **頂点 `i`** に対応する前提で、JSON では行ごとのベクタのリストに変換する。頂点 `1..n` が JSON 配列の **先頭から `n` 要素**になり、Python で読むときの行添字 `0..n-1` と自然に対応する（行の並びはオフセット変換を要しない）。

### 3.4 `save_catlab_graph_json(path, g; x=nothing, pretty=false)`

辞書を構築し、`JSON3.write` または `JSON3.pretty` でファイルに書き込む。

### 3.5 スクリプト直下の使用例

`PROGRAM_FILE` がこのファイル自身のときだけ実行されるブロックで、3 頂点・2 辺のグラフと 3×2 の特徴行列を例として `graph_from_catlab.json` に保存する。`include` して関数だけ使う場合は、このブロックをコメントアウトしてもよい。

---

## 4. Python 側: `import_catlab_json_to_pyg.py`

### 4.1 `catlab_json_to_data(path, *, dtype_edge_index=torch.int64)`

1. UTF-8 で JSON を読み、`format` が `catlab_directed_graph_v1` であることを検証する。
2. `num_nodes` を整数化する。
3. `edges` を `edge_index` に変換する。
   - 辺が無いときは `torch.empty((2, 0), dtype=...)`。
   - あるときは `torch.tensor(edges, dtype=...)` で `(num_edges, 2)` を作り、**`.t().contiguous()`** で `[2, num_edges]` にする。これが PyG が期待する **COO 形式**（1 行目が source、2 行目が target）である。
4. `Data(edge_index=edge_index, num_nodes=num_nodes)` を構築する。`num_nodes` を渡すことで、**辺が無い頂点や、`edge_index` の最大値だけでは決まらない頂点数**を正しく表現できる。
5. `x` キーがあれば `torch.tensor(..., dtype=torch.float32)` を `data.x` に代入する。

### 4.2 コマンドライン実行

引数に JSON パスを渡せる。省略時は同じディレクトリの `graph_from_catlab.json` を読む。

---

## 5. 1-based（Julia）を 0-based（Python）に揃える意図

### 5.1 言語・エコシステムの既定

- **Julia** および **Catlab の ACSet 上の頂点パート**は、配列アクセスと同様に **1 から始まる ID** が自然である（`add_edge!(g, 1, 2)` のように書く）。
- **PyTorch / PyG** の `edge_index` は、テンソルの列が **0 から `num_nodes - 1` の頂点インデックス**を指すのが慣例である。公式ドキュメント・既存モデル・`neighbor` サンプリングなどもこの前提に沿っている。

JSON を **どちらか一方の慣習に固定**しないと、Python 側で毎回 `edge_index -= 1` のような補正が必要になり、二重変換や条件分岐のバグ（辺だけ直して `x` の行対応を誤る等）の温床になる。

### 5.2 中間形式として 0-based を選ぶ理由

1. **受け手が PyG である**ため、中間表現を PyG のセマンティクスに合わせると、Python コードが **そのまま `Data` を下流に渡せる**。
2. **単一の変換箇所**: オフセットは Julia の書き出し時（`src-1`, `tgt-1`）に一度だけ行い、JSON を読む他言語・他ツールでも「頂点 ID は 0-based」と一文で仕様を固定できる。
3. **`num_nodes` は添字ではない**ため変換不要であることと、`x` の行順が「頂点 0, 1, … の特徴」であることとを組み合わせると、**孤立頂点を含むグラフ**でも一貫して解釈できる。

### 5.3 注意点

- Julia 側でグラフを構築したあと **頂点の削除**（`rem_vertex!` 等）を行うと、残る頂点 ID が連続でない場合がある。Catlab の `src`/`tgt` はそのときも **実際の頂点 ID** を返すため、`-1` した値が **0-based の「ラベル」として連続するとは限らない**。PyG は通常、頂点 ID が `0..num_nodes-1` に圧縮されていることを期待する。頂点を詰め直していないグラフを書き出す場合は、**書き出し前に頂点を正規化する**か、Python 側で **リマッピング**する別パイプラインが必要になる。本スクリプトは **連続 ID `1:n` の典型的なグラフ**を想定している。
- 逆方向（PyG から Catlab へ）を行う場合は、読み込み時に **+1** して `add_edge!(g, s+1, t+1)` のように戻す必要がある。

---

## 6. 実行例（まとめ）

```bash
# Julia: JSON 生成
julia --project=. export_catlab_graph_json.jl

# Python: Data へ変換して確認
python import_catlab_json_to_pyg.py graph_from_catlab.json
```

ライブラリとして利用する場合は、Julia から `include("export_catlab_graph_json.jl")` して `save_catlab_graph_json` を呼び、Python から `from import_catlab_json_to_pyg import catlab_json_to_data` して `catlab_json_to_data("path.json")` を使う。

---

## 7. 関連ファイル

| ファイル | 役割 |
|----------|------|
| `export_catlab_graph_json.jl` | Catlab グラフ → JSON |
| `import_catlab_json_to_pyg.py` | JSON → `torch_geometric.data.Data` |
| `Project.toml` / `Manifest.toml` | Julia 依存（Catlab, JSON3） |
