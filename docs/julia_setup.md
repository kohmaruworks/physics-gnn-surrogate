# Julia の導入とプロジェクト環境（`--project=src/julia`）

`julia: command not found` は **Julia 本体が未インストール**の状態です。  
Python の venv のように「インタープリタ＋専用パッケージ集合」を切り分けるのが、Julia では次の 2 層です。

| 層 | 内容 |
| --- | --- |
| **Julia 本体** | 言語の実行ファイル `julia`（必須） |
| **プロジェクト環境** | `src/julia/Project.toml`（と必要なら `Manifest.toml`）を `--project=...` で指す。`Pkg.instantiate()` で依存を固定 |

つまり **「仮想環境」＝`src/julia` ディレクトリ**として既に分離可能です。先に **Julia をユーザ領域に入れる**手順を取ってください（システム全体の `sudo snap` は必須ではありません）。

---

## 推奨: juliaup（公式に近い導入・複数版の切替）

Linux / WSL2:

```bash
curl -fsSL https://install.julialang.org | sh
```

案内に従い、パスを通す。新しいシェルで:

```bash
juliaup status
julia +release --version
```

初回の依存解決（**このリポのルート**で）:

```bash
cd /path/to/physics-gnn-surrogate-phase2
julia --project=src/julia -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` が無い場合は `resolve` 相当で解決され、必要に応じて `Pkg.update()` で固定できます。

スクリプト実行例:

```bash
julia --project=src/julia src/julia/03_simulation.jl
```

---

## 代替: 公式 tar（juliaup 不可の場合）

1. <https://julialang.org/downloads/> から **Linux x64** の最新安定版を取得  
2. 展開先に `export PATH=.../julia-*/bin:$PATH`  
3. 上と同じ `Pkg.instantiate()` を実行

---

## 代替: Snap（WSL/環境で案内が出る場合）

```bash
sudo snap install julia
```

バージョンがやや遅いことがあるため、**再現性が必要なら juliaup か公式 tar** を推奨します。

---

## act / basic と揃えた開発フロー

`physics-gnn-surrogate-act` / `physics-gnn-surrogate-basic` と同様、**Python は `.venv` で切り、パッケージは `uv pip install`**（`site-packages/*/INSTALLER` が `uv`）、**Julia は juliaup（システム側）**です。venv の中に Julia は入りません。

```bash
cd physics-gnn-surrogate-phase2
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
source scripts/phase2_env.sh   # venv を再度有効化しつつ juliaup を PATH に（README 参照）
julia --project=src/julia -e 'using Pkg; Pkg.instantiate()'
```

## Docker だけ使う（ホストに Julia を入れない）

リポジトリ内の `Dockerfile` を参照。ビルド後にコンテナ内で `instantiate` と実行が完結します。

```bash
cd physics-gnn-surrogate-phase2
docker build -t phase2-julia .
docker run --rm -v "$PWD":/app -w /app phase2-julia
```

`Dockerfile` の `CMD` を用途に合わせて差し替え可能です。

---

## よくある確認

- **`--project=src/julia` は仮想環境の指定**（そのディレクトリの `Project.toml` を有効化）  
- パッケージのソース・バイナリの大部分は `~/.julia`（プロジェクト専用ではないグローバルストア）。バージョンの固定用に **`src/julia/Manifest.toml` を `Pkg` が生成**する。再現性を上げるなら Git に含める。  
- REPL からは `julia` を起動し `]` で pkg モードのあと `activate src/julia` → `instantiate` でも同じ。

```bash
julia --project=src/julia -e 'using Pkg; Pkg.instantiate(); Pkg.status()'
```
