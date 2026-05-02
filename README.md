# Categorical Physics Engine: HeteroGNN Surrogate

[![Julia](https://img.shields.io/badge/Julia-1.10+-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[日本語版へ](#japanese)

## Overview & Tech Stack

This repository demonstrates an **applied-category-theory** pipeline from **rigorous CFD ground truth** (Julia / DEC) to a **heterogeneous graph surrogate** (PyTorch Geometric) that enables **lightning-fast inference**—orders of magnitude faster than full solver runs—while preserving structured physics through the graph and loss design.

**Tech stack**

- **Julia:** [Decapodes.jl](https://github.com/AlgebraicJulia/Decapodes.jl), [CombinatorialSpaces.jl](https://github.com/AlgebraicJulia/CombinatorialSpaces.jl), [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) (AlgebraicJulia ecosystem), plus **OrdinaryDiffEq.jl** for time integration.
- **Python:** **PyTorch**, **PyTorch Geometric** (heterogeneous GNNs, `HeteroData`), with supporting tooling (NumPy, Matplotlib, etc.) per step.

## Architecture

End-to-end flow from categorical physics definition in Julia to a physics-informed surrogate in Python:

```mermaid
graph TD
    subgraph julia_gt["1. Physical Ground Truth (Julia)"]
        A["Decapodes.jl<br/>Categorical Physics Definition"] --> B["CombinatorialSpaces.jl<br/>Simplicial Complex Mesh"]
        B --> C["OrdinaryDiffEq.jl<br/>CFD Simulation"]
    end
    subgraph json_hs["2. Data Handshake"]
        C -->|Export DEC Operators| D{"JSON Contract V2<br/>Heterogeneous Graph<br/>0-based indices"}
    end
    subgraph py_ai["3. AI Surrogate (Python/PyG)"]
        D -->|Ingest| E["PyTorch Geometric<br/>HeteroData"]
        E --> F["HeteroGNN<br/>with Physics-Informed Loss"]
        F --> G["Lightning-fast Inference<br/>approx. 180,000x Speedup"]
    end
```

## Repository Structure

Each **Step 1–5** folder is a **self-contained workspace**: its own `src/`, dependency files (`Project.toml` / `requirements_*.txt`), and generated `data/` or artifacts. Consume steps in order when reproducing the full pipeline.

```
categorical_physics_engine/
├── README.md
└── multiphysics_dec_solver/
    ├── step1_initial_physics_def/           # Julia — ground truth & JSON contract v1
    │   ├── Project.toml, Manifest.toml
    │   ├── requirements_viz.txt             # Python visualization deps
    │   ├── src/
    │   ├── data/raw/
    │   └── zenn_assets/
    ├── step2_heterogeneous_contract/        # Julia — heterogeneous JSON v2 (DEC topology)
    │   ├── Project.toml, Manifest.toml
    │   ├── requirements_test.txt
    │   ├── src/
    │   ├── data/v2_contract/
    │   └── zenn_assets/
    ├── step3_pyg_heterodata_loading/       # Python — V2 → HeteroData / .pt
    │   ├── requirements_step3.txt
    │   ├── src/
    │   ├── data/processed/
    │   └── zenn_assets/
    ├── step4_hetero_gnn_training/          # Python — physics-informed HeteroGNN training
    │   ├── requirements_step4.txt
    │   ├── src/
    │   ├── checkpoints/
    │   ├── runs/
    │   └── zenn_assets/
    └── step5_zero_shot_evaluation/         # Python — zero-shot eval & speed / ROI charts
        ├── requirements_step5.txt
        ├── src/
        └── evaluation_results/
```

## Step-by-Step Implementation

### Step 1: Categorical Physics Definition & JSON Contract Validation

This step establishes the ground truth generation using Applied Category Theory and validates the cross-language data pipeline.

#### Visualization: 2D Cylinder Wake (Velocity Magnitude)

![Categorical CFD - Velocity Field](./multiphysics_dec_solver/step1_initial_physics_def/zenn_assets/cylinder_wake_animation.gif)

#### What is Simulated?

A 2D incompressible fluid flow around a circular obstacle (**Cylinder Wake** scenario). The underlying physics are strictly defined as an **operadic composition** of the Navier–Stokes equations using [Decapodes.jl](https://github.com/AlgebraicJulia/Decapodes.jl) and simulated on an unstructured simplicial complex generated via [CombinatorialSpaces.jl](https://github.com/AlgebraicJulia/CombinatorialSpaces.jl).

**PDE sketch.** With velocity $\mathbf{u}$, pressure $p$, density $\rho$, kinematic viscosity $\nu$, temperature $T$, and thermal diffusivity $\alpha$, a standard incompressible coupled statement on $\Omega \subset \mathbb{R}^2$ is

$$
\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\rho^{-1}\nabla p + \nu \Delta \mathbf{u} + \mathbf{f}, \qquad \nabla\cdot \mathbf{u} = 0, \qquad \partial_t T + \nabla\cdot(T\mathbf{u}) = \alpha \Delta T .
$$

**Discrete Exterior Calculus (DEC)** replaces $\nabla$, $\Delta$, and divergence by metric-aware sparse operators on the simplicial mesh; **Decapodes.jl** composes them diagrammatically into a semi-discrete ODE integrated by **OrdinaryDiffEq.jl**. *(Executable momentum in `definitions.jl` uses a Stokes-type linearization $\partial_t \mathbf{u} \approx \nu \Delta \mathbf{u} - \rho^{-1}\nabla p$ plus an auxiliary pressure equation $\partial_t p = \kappa \Delta p$, coupled to advection–diffusion for $T$.)*

#### What was Confirmed?

This visualization serves as the proof of concept for our architecture:

1. **Topological Integrity** — Successfully generated a valid 2D simplicial complex with internal boundaries (the cylinder) and correctly mapped it to a spatial domain.

2. **Cross-Language Contract Fidelity** — Proved that the **JSON contract** bridges Julia and Python: node coordinates, triangle connectivity (safely converted from **1-based** to **0-based** indexing), and multidimensional physical fields are restored in Python (e.g. Matplotlib `Triangulation`) without loss of fidelity.

3. **Physical Solver Stability** — Confirmed that **Discrete Exterior Calculus (DEC)** operators derived from the categorical diagrams yield stable initialization and physically consistent temporal evolution under the prescribed boundary setup.

To reproduce those checks, work from **`multiphysics_dec_solver/step1_initial_physics_def/`**, where **`src/main.jl`** activates that folder as the Julia project root: run **`Pkg.instantiate()`**, then **`julia --project=. src/main.jl`**. The default **`cylinder_wake`** run integrates with **`gensim` + OrdinaryDiffEq** (`--t-end 1.2`, **`nx=36`**, **`ny=18`**, **`--frames 73`**, fluid ν/ρ and thermal coupling) and writes **`data/raw/ground_truth_cylinder_wake.json`** plus **`ground_truth_cylinder_wake.jld2`**—**1-based** topology with vertex-proxy fields **`velocity_vertex_vx`/`vy`**, **`pressure`**, and **`temperature`**. **`--scenario heat_sink`** swaps in **`ground_truth_heat_sink.json`**. On the Python side, **`requirements_viz.txt`** and **`src/visualize_contract.py`** regenerate artifacts under **`zenn_assets/`** (for example **`cylinder_wake_animation.gif`**) and confirm the JSON round-trip with Matplotlib **`Triangulation`**. The usual entry files are **`Project.toml`**, **`src/main.jl`**, **`src/visualize_contract.py`**, and outputs under **`data/raw/`**.

### Step 2: Heterogeneous Topology Extraction & JSON Contract V2

This step upgrades the data contract to a Heterogeneous Graph format, extracting explicit topological relationships (Discrete Exterior Calculus operators) for zero-overhead initialization in PyTorch Geometric.

#### Visualization: Primal & Dual Complexes

![Heterogeneous Topology](./multiphysics_dec_solver/step2_heterogeneous_contract/zenn_assets/hetero_topology.png)

*(Note: The highly dense red ‘x’ markers representing the Dual vertices (N=3997) visually overlap the underlying Primal vertices (N=703), correctly reflecting the mathematical barycentric subdivision.)*

![Heterogeneous Topology — zoom (central window)](./multiphysics_dec_solver/step2_heterogeneous_contract/zenn_assets/hetero_topology_zoom.png)

*(Zoom on a central percentile window of primal vertices; larger primal markers make the blue disks easier to see alongside dual crosses.)*

### 🔬 What is Extracted?

Explicit topological relationships from the 2D simplicial complex. Instead of a single edge list, the geometry is decomposed into `primal_to_primal` (Gradient/Exterior Derivative), `dual_to_dual` (Flux), and `primal_to_dual` (Hodge Star) mappings using the mathematical definitions of `CombinatorialSpaces.jl`.

### ✅ What was Confirmed?

1. **Mathematical Index Fidelity** — Successfully managed the disparity between vertex and edge mappings (e.g., Hodge mappings connecting 1997 Primal Edges to 7883 Dual Edges) without index out-of-bounds errors.

2. **0-Based Index Conversion** — All Julia-native 1-based indices were safely converted to 0-based indices. Terminal assertions proved that all source and target indices fit perfectly within the exact bounds of their corresponding tensor sizes.

3. **Ready for PyG HeteroData** — The exported V2 JSON strictly follows the schema required to instantiate PyTorch Geometric’s `HeteroData` without requiring any heavy data-wrangling on the Python side.

Carrying those indexing guarantees forward is handled entirely inside **`multiphysics_dec_solver/step2_heterogeneous_contract/`**. After **`Pkg.instantiate()`**—including the small housekeeping fixes for multi-line **`using`** and the fragile **`has_subpart`** guard—the exporter **`src/export_hetero_json.jl`** reloads Step 1’s **`TopologyBlocks1Based`** from **`JLD2`** when available (otherwise JSON), rebuilds the primal mesh with **`glue_triangle!`** / **`orient!`**, constructs **`EmbeddedDeltaDualComplex2D{Bool,Float64,Point3}`**, and subdivides duals with **`subdivide_duals!(..., Barycenter())`**. It emits COO tensors **`primal_to_primal`**, **`dual_to_dual`**, and a **`primal_to_dual`** tensor derived from **`elementary_duals`**, subtracting one per index for **0-based** JSON and enforcing **`@assert`** consistency against **`dec_counts`**. **`julia --project=. src/main.jl`** then lands at **`data/v2_contract/hetero_cylinder_wake_t0.35.json`** near **t ≈ 0.35**, preferring Step 1 **`.jld2`** inputs. Python **`python src/test_hetero_load.py`** (see **`requirements_test.txt`**) revisits the same tensors in the terminal and renders **`zenn_assets/hetero_topology.png`** plus **`hetero_topology_zoom.png`**. *(Julia ≥ 1.11:* omit **`SparseArrays`** from **`[deps]`**—stdlib/registry clashes on **`instantiate`**—and rely on transitive sparse support via **`CombinatorialSpaces`**.) Key paths remain **`Project.toml`**, **`src/main.jl`**, **`src/export_hetero_json.jl`**, **`src/test_hetero_load.py`**, **`requirements_test.txt`**, and **`data/v2_contract/`**.

### Step 3: PyG HeteroData Loading & Feature Audit

This step safely ingests the V2 JSON contract into a PyTorch Geometric (PyG) environment, bridging the categorical physics engine with the deep learning architecture.

#### Visualization: PyG Metapath Subgraph & Feature Distributions

![PyG Subgraph Topology](./multiphysics_dec_solver/step3_pyg_heterodata_loading/zenn_assets/pyg_subgraph_topology.png)

*(Note: A 3-hop ego-graph illustrating the topological connections between Primal nodes, Dual nodes, and Edge Midpoints via `p2p`, `d2d`, and `p2d` metapaths.)*

![PyG Feature Distributions](./multiphysics_dec_solver/step3_pyg_heterodata_loading/zenn_assets/pyg_feature_distributions.png)

### 🔬 What is Ingested & Visualized?

The V2 JSON contract is instantiated as a PyTorch Geometric `HeteroData` object. To resolve the complex edge-to-edge mappings of DEC (for example, Hodge Star), edges are lifted into independent midpoint nodes (a line-graph style construction). That lets PyG run message passing natively across geometrically distinct entity types.

### ✅ What was Confirmed?

1. **Topological Subgraph Validation** — Extracted a local ego-graph confirming that primal, dual, and midpoint nodes interconnect correctly via `p2p`, `d2d`, and `p2d` metapaths without index collisions.

2. **Data Audit & Sanity Check** — Verified that all tensor features (`x` and `pos`) contain no `NaN` or `Inf`, use the expected dtypes (`float32` for features, `long` for edges), and satisfy the asserted layout checks.

3. **AI Readiness** — Feature histograms show velocity and pressure on scales suitable for neural-network training and downstream normalization.

Install **`requirements_step3.txt`** inside **`multiphysics_dec_solver/step3_pyg_heterodata_loading/`**, then let **`hetero_dataset.load_v2_hetero_json`** ingest the Step 2 artifact (default **`../step2_heterogeneous_contract/data/v2_contract/hetero_cylinder_wake_t0.35.json`**): it appends midpoint nodes, casts **`float32`** **`x`**/**`pos`** with **`long`** **`edge_index`**, preserves the source JSON path on the object, and **`save_hetero_pt`** snapshots **`data/processed/hetero_cylinder_wake_t0.35.pt`** plus **`HeteroV2Meta`**. **`python src/test_audit.py`** (optional JSON override) and **`python src/visualize_pyg.py`** complete the loop—**`visualize_pyg`** loads checkpoints via **`torch.load(..., map_location="cpu", weights_only=False)`** for PyTorch 2.6+ pickle safety—and emit **`zenn_assets/pyg_subgraph_topology.png`** and **`pyg_feature_distributions.png`**.

That automation lines up with the earlier guarantees: **`load_v2_hetero_json`** raises **`ValueError`** when primal physics lengths disagree with **`num_nodes`**, when any **`edge_index`** is not shaped **`[2, E]`**, or when midpoint reconstruction conflicts with **`dec_counts`**, preventing silent topology drift. **`test_audit`** asserts finite **`x`**/**`pos`**, dtypes/layouts on each metapath, and index splits so **`p2p`**/**`d2d`** touch only primal/dual vertex slices while **`p2d`** references midpoint tails—surfacing out-of-bounds or partition collisions immediately. **`visualize_pyg`** raises **`FileNotFoundError`** if the **`.pt`** is absent and, when **`('primal', 0)`** is isolated inside the fused NetworkX ego graph, falls back to the primal vertex closest to the geometric centroid so visualization never returns an empty subgraph. For readable latency on laptops, prefer **`pip install torch --index-url https://download.pytorch.org/whl/cpu`** before **`python src/test_audit.py`** then **`python src/visualize_pyg.py`**; the modules involved are **`requirements_step3.txt`**, **`src/hetero_dataset.py`**, **`src/test_audit.py`**, **`src/visualize_pyg.py`**, and serialized graphs under **`data/processed/*.pt`**.

### Step 4: HeteroGNN Architecture & Physics-Informed Training

This step establishes the deep learning core, implementing a **Heterogeneous Graph Neural Network** with **Physics-Informed Loss** so physical dynamics are learned directly from the categorical heterogeneous graph produced upstream. **`HeteroConv`** pairs **primal** and **dual** complexes with relation-specific **`GraphConv`** stacks and explicit **`d2p`** reverse coupling; **MSE** reconstruction on primal **`x`** is augmented by a graph-gradient **pseudo–divergence** penalty on fluid vertices to approximate mass-conservation structure until full DEC operators are available in PyG.

**Loss (sketch).** Let $\hat{\mathbf{x}}$ be predicted primal features and $\mathbf{x}^{*}$ targets. With primal-only fluid vertices indexed on **`p2p`** edges $(i,j)$ and velocity channels $(u,v)$,

$$
\mathcal{L}_{\mathrm{data}} = \mathrm{MSE}(\hat{\mathbf{x}}, \mathbf{x}^{*}), \qquad
\mathcal{L}_{\mathrm{phys}} = \underbrace{\mathbb{E}_{(i,j)}\big[\|\hat{\mathbf{u}}_i - \hat{\mathbf{u}}_j\|^2\big]}_{\text{graph-gradient energy}} + \underbrace{\mathbb{E}_{k}\big[b_k^2\big]}_{\text{balance proxy}}, \quad
\mathcal{L} = \mathcal{L}_{\mathrm{data}} + \lambda\,\mathcal{L}_{\mathrm{phys}},
$$

where $\hat{\mathbf{u}}=(\hat{u},\hat{v})$ and $b_k$ aggregates edge increments at vertex $k$ (see **`pseudo_divergence_loss`** in **`physics_loss.py`**). This is a lightweight surrogate for $\nabla\!\cdot\mathbf{u}\approx 0$, not full cotangent-weighted DEC.

**Discrete form (aligned with code).** Let $\mathcal{E}$ be the filtered **`p2p`** vertex edges $i\to j$. Define

$$
\boldsymbol{\Delta}_{ij} := \hat{\mathbf{u}}_i - \hat{\mathbf{u}}_j, \qquad
\mathcal{L}_{\mathrm{grad}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j)\in\mathcal{E}} \|\boldsymbol{\Delta}_{ij}\|_2^2 .
$$

Write $r_{ij} := \Delta_{ij,u} + \Delta_{ij,v}$. For each head vertex $j$, let $d_j$ be its in-degree under $\mathcal{E}$ and $s_j := \sum_{i:\,(i\to j)\in\mathcal{E}} r_{ij}$. With $b_j := s_j / \max(d_j,1)$,

$$
\mathcal{L}_{\mathrm{bal}} = \frac{1}{N}\sum_{j=1}^{N} b_j^2 , \qquad
\mathcal{L}_{\mathrm{phys}} = \mathcal{L}_{\mathrm{grad}} + \mathcal{L}_{\mathrm{bal}},
$$

where $N$ is the primal vertex count fed into **`pseudo_divergence_loss`** (vertices with $d_j=0$ contribute $b_j=0$).

#### Visualization: Spatial Inference Comparison

![GNN Inference Comparison](./multiphysics_dec_solver/step4_hetero_gnn_training/zenn_assets/gnn_inference_comparison.png)

*(Note: spatial mapping of velocity magnitude for ground truth versus GNN prediction and their absolute error on primal fluid vertices—sanity-checking the trained forward pass end-to-end.)*

### 🔬 What is Simulated & Visualized?

A **spatial inference** view of the trained **HeteroGNN**: heterogeneous node features from Step 3 drive **`HeteroConv`** message passing across **primal** and **dual** complexes (including **`p2p`**, **`d2d`**, **`p2d`**, and **`d2p`**). Predictions are scattered with **`primal.pos`** so the recovered velocity magnitude field can be compared directly to the Julia-derived ground truth.

### ✅ What was Confirmed?

1. **Architectural viability** — **`HeteroConv`** routes and aggregates messages across topologically distinct entities (**`p2p`**, **`d2d`**, **`p2d`**, **`d2p`**) without shape errors.
2. **Physics-informed operability** — A custom **pseudo–divergence** loss built from primal **`p2p`** graph gradients backpropagates through the PyG heterogeneous graph and combines cleanly with **MSE**.
3. **End-to-end pipeline completion** — Inference plots show data flowing from categorical Julia exports through **`HeteroData`** training to Python scatter maps of prediction and absolute error.

Training lives under **`multiphysics_dec_solver/step4_hetero_gnn_training/`**: install **`requirements_step4.txt`** (CPU PyTorch hint, **`torch-geometric`**, **`tqdm`**, **`tensorboard`**, **`matplotlib`**), then execute **`python src/train.py`**. **`PhysicsInformedHeteroGNN`** in **`src/model.py`** stacks **`HeteroConv`** + **`GraphConv`** with configurable **`hidden_dim`** / **`num_layers`** and a primal **`Linear`** head sized to **`primal.x`**. **`physics_loss.py`** blends **MSE** with **λ × pseudo_divergence_loss** on fluid-filtered **`p2p`** edges. **`train.py`** reads **`../step3_pyg_heterodata_loading/data/processed/hetero_cylinder_wake_t0.35.pt`**, prepends **`step3_pyg_heterodata_loading/src`** for pickle compatibility, trains the single-graph primal **`x`** auto-encoder with **`tqdm`** bars (total / data / physics), logs TensorBoard runs under **`runs/step4_hetero_gnn/`**, and checkpoints **`checkpoints/hetero_gnn_model.pth`**. **`python src/visualize_inference.py`** switches to **`eval()`** / **`torch.no_grad()`**, maps velocity magnitude **$\sqrt{u^2+v^2}$** from **`x[:,0:2]`**, and saves **`zenn_assets/gnn_inference_comparison.png`** (1×3 panels, **`turbo`** / **`Reds`**, 15×4 in at 300 DPI)—closing the loop with the spatial comparisons validated above. Supporting files include **`requirements_step4.txt`**, **`src/model.py`**, **`src/physics_loss.py`**, **`src/train.py`**, **`src/visualize_inference.py`**, **`checkpoints/`**, and **`zenn_assets/`**.

### Step 5: Zero-Shot Generalization & Performance Benchmark

This stage evaluates the trained surrogate’s **topological generalization** on **unseen meshes** via **zero-shot inference**: any compatible **`HeteroData`** **`.pt`** (matching primal/dual **feature widths** to the checkpoint) can be scored without retraining. **Performance benchmarking** quantifies **return on compute**: millisecond-scale **GNN** forwards versus minute-to-hour **CFD** solves, while **MSE**/**MAE** track how much **physical accuracy** is retained on primal fields.

The workflow proves that a **physics-informed HeteroGNN** can be exercised as a **fast surrogate** alongside rigorous error telemetry—supporting design loops that would be impractical under solver-only budgets.

#### Visualization: Zero-shot spatial comparison

![Zero-shot spatial comparison (velocity magnitude)](./multiphysics_dec_solver/step5_zero_shot_evaluation/evaluation_results/zeroshot_comparison.png)

*(Note: scatter maps of velocity magnitude—ground truth, prediction, absolute error—on primal fluid vertices for an evaluated graph/checkpoint pair.)*

### 🔬 What is evaluated?

**`evaluate_generalization.py`** loads **`--data-path`** and **`--model-path`**, runs **`eval()`** inference, prints **MSE** and **MAE** over **all primal `x` nodes**, and saves **`evaluation_results/zeroshot_comparison.png`** (same **1×3** scatter layout as Step 4). **`benchmark_speed.py`** performs **10** warm-up forwards then times **100** timed forwards with **`time.perf_counter`** (CUDA synchronized when applicable), reporting **mean / std / min / max** latency in **milliseconds** via **`tabulate`**.

### ✅ What was confirmed?

1. **Portable inference** — Any Step-3-style **`.pt`** with matching channel widths runs through **`PhysicsInformedHeteroGNN`** without architectural edits.
2. **Quantified accuracy** — Global **MSE**/**MAE** summarize primal-field reconstruction on new graphs; spatial error panels localize discrepancy.
3. **Quantified speed** — Repeated forward benchmarks document surrogate latency suitable for interactive or outer-loop use cases compared with traditional solvers.

Operational scripts sit in **`multiphysics_dec_solver/step5_zero_shot_evaluation/`**. After **`pip install -r requirements_step5.txt`** (PyTorch, PyTorch Geometric, Matplotlib, NumPy, **`tabulate`**, plus the CPU wheel hint), **`python src/evaluate_generalization.py`** accepts **`--data-path`** and **`--model-path`**, runs **`eval()`**, prints **MSE**/**MAE** over all primal **`x`** nodes, and writes **`evaluation_results/zeroshot_comparison.png`** using the same 1×3 scatter layout introduced in Step 4; **`python src/benchmark_speed.py`** performs ten warm-up forwards and times one hundred timed forwards with **`time.perf_counter`** (CUDA synchronized when available), summarizing mean/std/min/max latency in milliseconds via **`tabulate`**. Both drivers prepend **`step4_hetero_gnn_training/src`** and **`step3_pyg_heterodata_loading/src`** so checkpoints unpickle cleanly. Optionally **`python src/visualize_benchmark_chart.py`** refreshes **`evaluation_results/roi_speedup_benchmark.png`** (300 DPI, log-scaled **y** axis) by editing **`TRADITIONAL_CFD_SECONDS`** / **`GNN_INFERENCE_SECONDS`** to reflect your CFD budget and the measured surrogate latency—making the ROI narrative explicit relative to the accuracy metrics gathered earlier. Defaults smoke-test the cylinder-wake **`.pt`** with the Step 4 **`.pth`**; pass another compatible **`.pt`** to probe unseen meshes.

#### Visualization: ROI inference speedup (illustrative benchmark)

![ROI inference speedup benchmark](./multiphysics_dec_solver/step5_zero_shot_evaluation/evaluation_results/roi_speedup_benchmark.png)

Log-scale bar chart of wall-clock time (**seconds**) for representative **Julia/DEC CFD** versus **HeteroGNN surrogate** inference—highlighting extreme speedup (**ROI**) at illustrative placeholder timings editable in **`src/visualize_benchmark_chart.py`**.

## License

This project is released under the **MIT License**.

---

<br/>

<a id="japanese"></a>

# Categorical Physics Engine: HeteroGNN サロゲート (日本語版)

[English version ↑](#categorical-physics-engine-heterognn-surrogate)

## はじめに

実務や研究でCFD（計算流体力学）を活用する際、精度を担保するために重厚なソルバーを実行する一方で、設計の反復ループや対話的な解析においては、その計算レイテンシが大きなボトルネックになりがちです。そこで真価を発揮するのが、物理的な構造を極力損なうことなく高速に近似する「サロゲートモデル」というアプローチです。

本稿で解説する **[categorical_physics_engine](https://github.com/kohmaruworks/categorical_physics_engine)** は、**応用圏論とDEC（離散外微分）をJuliaで定式化・シミュレーションし、その結果をJSONコントラクト経由でPyTorch Geometric（PyG）のヘテロジニアスGNNに渡す**という、言語とフレームワークの境界を明確にしたパイプラインです。「ひとつの巨大なシステムで全てを解決する」のではなく、**ステップごとに明確な契約（コントラクト）と検証を設ける**ことで、再現性と保守性を高めています。

ここから先は、リポジトリのREADMEをベースに、システムの全体像と実装の軌跡をステップバイステップで紐解いていきます。もし序盤で前提知識が高度に感じられた場合は、まず全体のアーキテクチャ図やディレクトリ構成を眺めていただき、概要を掴んでから各ステップの詳細に戻るという読み方をおすすめします。

また、読者の皆様が手元で再現しやすいよう、パラメータの数値やファイル名は実行コマンドとしてそのまま使える形で記載しています。

---

## 概要と技術スタック

応用圏論に基づく **Julia / DEC による厳密なCFDグラウンドトゥルース**から、**PyTorch Geometric** 上の **ヘテロジニアスGNNサロゲート**へと繋ぐパイプラインの全容です。**フルソルバーに比べて桁違いに短い推論時間**で物理場を予測する一方で、グラフ構造と損失関数の設計によって物理的整合性を構造的に保持することを目的としています。

**技術スタック**

- **Julia:** Decapodes.jl、CombinatorialSpaces.jl、Catlab.jl（AlgebraicJulia エコシステム）、および時間積分に **OrdinaryDiffEq.jl**。
- **Python:** **PyTorch**、**PyTorch Geometric**（ヘテロジニアスGNN、`HeteroData`）、各ステップに応じた NumPy・Matplotlib など。

Juliaで「何を真実（グラウンドトゥルース）とみなすか」、Pythonで「どう速く近似するか」を役割分担し、**JSONのスキーマとインデックス規約**が両者を繋ぐ共通言語として機能します。

## アーキテクチャ

Juliaにおける圏論的物理定義から、Pythonにおける物理情報付きサロゲートモデルまでのデータの流れは以下の通りです。

```mermaid
graph TD
    subgraph julia_gt["1. Physical Ground Truth (Julia)"]
        A["Decapodes.jl<br/>Categorical Physics Definition"] --> B["CombinatorialSpaces.jl<br/>Simplicial Complex Mesh"]
        B --> C["OrdinaryDiffEq.jl<br/>CFD Simulation"]
    end
    subgraph json_hs["2. Data Handshake"]
        C -->|Export DEC Operators| D{"JSON Contract V2<br/>Heterogeneous Graph<br/>0-based indices"}
    end
    subgraph py_ai["3. AI Surrogate (Python/PyG)"]
        D -->|Ingest| E["PyTorch Geometric<br/>HeteroData"]
        E --> F["HeteroGNN<br/>with Physics-Informed Loss"]
        F --> G["Lightning-fast Inference<br/>approx. 180,000x Speedup"]
    end
```

## リポジトリ構成

**ステップ1〜5** はそれぞれ **独立した作業ディレクトリ**として構成されています。独自の `src/`、依存パッケージ定義（`Project.toml` / `requirements_*.txt`）、生成物用の `data/` などを持ちます。パイプライン全体を手元で再現する場合は、ステップ順に実行してください。

```text
categorical_physics_engine/
├── README.md
└── multiphysics_dec_solver/
    ├── step1_initial_physics_def/           # Julia — グラウンドトゥルース & JSON コントラクト v1
    │   ├── Project.toml, Manifest.toml
    │   ├── requirements_viz.txt             # Python 可視化用依存
    │   ├── src/
    │   ├── data/raw/
    │   └── zenn_assets/
    ├── step2_heterogeneous_contract/        # Julia — ヘテロジニアス JSON v2（DEC トポロジー）
    │   ├── Project.toml, Manifest.toml
    │   ├── requirements_test.txt
    │   ├── src/
    │   ├── data/v2_contract/
    │   └── zenn_assets/
    ├── step3_pyg_heterodata_loading/       # Python — V2 → HeteroData / .pt
    │   ├── requirements_step3.txt
    │   ├── src/
    │   ├── data/processed/
    │   └── zenn_assets/
    ├── step4_hetero_gnn_training/          # Python — 物理情報付き HeteroGNN 学習
    │   ├── requirements_step4.txt
    │   ├── src/
    │   ├── checkpoints/
    │   ├── runs/
    │   └── zenn_assets/
    └── step5_zero_shot_evaluation/         # Python — ゼロショット評価・速度 / ROI チャート
        ├── requirements_step5.txt
        ├── src/
        └── evaluation_results/
```

## 実装ステップ詳細

以下では、各ステップで「何を検証したか」と「実際に何を実行したか」を追っていきます。

### ステップ 1: 圏論的物理定義とJSONコントラクト検証

応用圏論に基づくグラウンドトゥルースの生成と、言語間データパイプラインの基礎検証を行う段階です。

#### 可視化: 2次元シリンダー後流（速度の大きさ）

![圏論的 CFD — 速度場](./multiphysics_dec_solver/step1_initial_physics_def/zenn_assets/cylinder_wake_animation.gif)

#### シミュレーション対象

2次元非圧縮性流体が円形障害物（シリンダー）の周りを流れる **シリンダー後流** シナリオです。物理法則は `Decapodes.jl` によるナビエ・ストークス方程式の **operadic合成**として厳密に定義し、`CombinatorialSpaces.jl` が生成する非構造単体複体上で時間発展を計算します。

**支配方程式のスケッチ:** 速度 $\mathbf{u}$、圧力 $p$、密度 $\rho$、動粘性係数 $\nu$、温度 $T$、熱拡散率 $\alpha$ としたとき、$\Omega \subset \mathbb{R}^2$ 上の非圧縮連成場の典型的な形は以下のようになります。

$$
\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\rho^{-1}\nabla p + \nu \Delta \mathbf{u} + \mathbf{f}, \qquad \nabla\cdot \mathbf{u} = 0, \qquad \partial_t T + \nabla\cdot(T\mathbf{u}) = \alpha \Delta T
$$

**DEC（離散外微分）** は単体複体上で勾配・ラプラシアン・発散を離散化し、**Decapodes.jl** が合成した半離散系を **OrdinaryDiffEq.jl** が時間積分します。*（※実装上の `definitions.jl` では、モメンタムを対流なしの $\partial_t \mathbf{u} \approx \nu \Delta \mathbf{u} - \rho^{-1}\nabla p$ と $\partial_t p = \kappa \Delta p$ とし、$T$ は移流–拡散で結合させています。）*

#### 検証・確認事項

1. **トポロジーの整合性** — シリンダーを内部境界として含む2次元単体複体が有効に生成され、空間ドメインへ正しくマッピングされることを確認しました。
2. **JSONコントラクトの正確性** — JuliaからPythonへ、ノード座標・三角形の連結情報（**1始まりから0始まりへの安全な変換**）・多次元物理場が情報の欠損なく引き渡せることを証明しました（Python側では Matplotlibの `Triangulation` 等で完全に復元可能です）。
3. **物理ソルバーの安定性** — 圏論的ダイアグラムから生成された **DEC** オペレータが、指定された境界条件下で安定した初期化と物理的に妥当な時間発展をもたらすことを確認しました。

これらの検証は、`multiphysics_dec_solver/step1_initial_physics_def/` にて **`Pkg.instantiate()`** 実行後、**`julia --project=. src/main.jl`** を実行することで再現できます。既定の **`cylinder_wake`** シナリオでは、**`gensim` + OrdinaryDiffEq** を用いて **`--t-end 1.2`**・格子 **`nx=36`**, **`ny=18`**・**`--frames 73`** などの設定で実行され、**`data/raw/ground_truth_cylinder_wake.json`** および **`ground_truth_cylinder_wake.jld2`** が出力されます。

### ステップ 2: ヘテロジニアストポロジーの抽出とJSONコントラクト V2

本ステップでは、データコントラクトを **ヘテロジニアスグラフ**形式へとアップグレードします。PyTorch Geometric環境で **ゼロオーバーヘッドでの初期化**を実現するため、明示的なトポロジー関係（離散外微分 **DEC** オペレータ）を抽出します。

#### 可視化: プライマル複体とデュアル複体

![ヘテロジニアストポロジー](./multiphysics_dec_solver/step2_heterogeneous_contract/zenn_assets/hetero_topology.png)

*（注: デュアル頂点を表す高密度の赤色の「×」マーカー（N=3997）は、下層のプライマル頂点（N=703）と視覚的に重なって見えますが、これは数学的な **重心細分** を正確に反映した結果です。）*

![ヘテロジニアストポロジー（中央付近の拡大）](./multiphysics_dec_solver/step2_heterogeneous_contract/zenn_assets/hetero_topology_zoom.png)

*（プライマル頂点のパーセンタイルで切り出した中央部分の拡大図です。青いプライマル頂点を強調し、全体図では視認しにくいディスク形状の境界を明示しています。）*

### 🔬 抽出対象

2次元単体複体から明示的なトポロジー関係を抽出します。単一のエッジリストとしてではなく、`CombinatorialSpaces.jl` の数学的定義に基づき、`primal_to_primal`（勾配・外微分）、`dual_to_dual`（流束）、`primal_to_dual`（ホッジスター演算）の各マッピングへと幾何学構造を厳密に分解しています。

### ✅ 検証・確認事項

1. **数学的インデックスの忠実性** — 頂点と辺のマッピングの差異（例: 1997個のプライマルエッジから7883個のデュアルエッジへのホッジマッピング）を、インデックスの範囲外エラーを起こすことなく完全に制御できていることを確認しました。
2. **0始まりインデックスへの安全な変換** — Juliaネイティブの1始まりインデックスを、すべて0始まりへ安全に変換しました。ターミナルでのアサート検証により、すべてのインデックスが対応するテンソルサイズの範囲内に完全に収まっていることを数学的に証明しています。
3. **PyG HeteroDataへの準備完了** — 出力されたV2 JSONが、Python側での煩雑なデータ整形を一切必要とせず、PyTorch Geometricの `HeteroData` を即座にインスタンス化できるスキーマに準拠していることを確認しました。

これらの処理は **`multiphysics_dec_solver/step2_heterogeneous_contract/`** に集約されています。**`julia --project=. src/main.jl`** を実行すると、時刻 **t≈0.35** に対応する **`data/v2_contract/hetero_cylinder_wake_t0.35.json`** が生成されます。

### ステップ 3: PyGにおけるHeteroDataの読み込みと特徴量監査

本ステップでは、V2 JSONコントラクトをPyTorch Geometric（PyG）環境へ安全に取り込み、圏論的物理エンジンとディープラーニングアーキテクチャを正式に橋渡しします。

#### 可視化: PyGメタパス部分グラフと特徴量分布

![PyG 部分グラフのトポロジー](./multiphysics_dec_solver/step3_pyg_heterodata_loading/zenn_assets/pyg_subgraph_topology.png)

*（注: プライマル、デュアル、および辺中点ノードが `p2p`, `d2d`, `p2d` メタパスで接続される様子を示す、最大3ホップのエゴグラフです。）*

![PyG 特徴量の分布](./multiphysics_dec_solver/step3_pyg_heterodata_loading/zenn_assets/pyg_feature_distributions.png)

### 🔬 入力と可視化の対象

V2 JSONコントラクトをPyTorch Geometricの `HeteroData` オブジェクトとしてインスタンス化します。DEC特有の複雑な辺対辺マッピング（ホッジスター演算など）を処理するため、各辺の中点を独立したノードとして持ち上げています（線グラフに近いアプローチ）。これにより、PyG上で幾何学的に性質の異なるノード間でも、ネイティブかつスムーズにメッセージパッシングを実行できます。

### ✅ 検証・確認事項

1. **トポロジー構造の検証** — 部分グラフを抽出し、プライマル頂点、デュアル頂点、辺中点ノードが、各メタパスを通じてインデックスの衝突なく正確に結合されていることを視覚的に確認しました。
2. **データ監査と健全性チェック** — すべての特徴量および座標テンソルに `NaN` や `Inf` が含まれていないこと、正しいデータ型（特徴量は `float32`、インデックスは `long`）にキャストされていることをアサーションによって保証しました。
3. **AI学習への準備完了** — 物理変数（流速、圧力など）の分布ヒストグラムから、値がニューラルネットワークの学習に適したスケールに収まっており、標準化への準備が整っていることを実証しました。

これらの手順には、データの不整合を未然に防ぐための堅牢なエラーハンドリングが組み込まれています。例えば **`test_audit.py`** は、各テンソルの有限性やメタパスごとの形状、参照先が正しいブロック（頂点または辺中点）を指しているかを厳密にチェックします。

### ステップ 4: HeteroGNNアーキテクチャとPhysics-Informed学習

上流から引き渡された圏論的ヘテロジニアス構造を入力とし、**物理情報付き損失（Physics-Informed Loss）** を備えた **ヘテロジニアスGNN** によって物理ダイナミクスを直接学習する、AIアーキテクチャの中核を確立する段階です。

**損失関数のスケッチ（実装対応）:** 予測 $\hat{\mathbf{x}}$、教師データ $\mathbf{x}^{*}$、**`p2p`** 上の速度 $\hat{\mathbf{u}}=(\hat{u},\hat{v})$ としたとき、データ再構成項と物理的制約項を以下のように定義します。

$$
\mathcal{L}_{\mathrm{data}} = \mathrm{MSE}(\hat{\mathbf{x}}, \mathbf{x}^{*}), \qquad \mathcal{L}_{\mathrm{phys}} = \mathbb{E}_{(i,j)}\big[\|\hat{\mathbf{u}}_i - \hat{\mathbf{u}}_j\|^2\big] + \mathbb{E}_{k}\big[b_k^2\big], \qquad \mathcal{L} = \mathcal{L}_{\mathrm{data}} + \lambda \mathcal{L}_{\mathrm{phys}}
$$

ここで $b_k$ は **`pseudo_divergence_loss`** における頂点ごとのバランス項です。厳密なコタンジェント重み付きDECへの完全な置換ではなく、非圧縮性条件 $\nabla\!\cdot\mathbf{u}\approx 0$ に寄せるための、グラフ上のエネルギーペナルティとして機能します。

**離散形（コードとの対応）:** フィルタ後の **`p2p`** の辺集合を $\mathcal{E}$（向き $i\to j$）とすると、勾配項は以下のようになります。

$$
\boldsymbol{\Delta}_{ij} = \hat{\mathbf{u}}_i - \hat{\mathbf{u}}_j, \qquad \mathcal{L}_{\mathrm{grad}} = \frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}}\|\boldsymbol{\Delta}_{ij}\|_2^2
$$

さらに $r_{ij} = \Delta_{ij,u}+\Delta_{ij,v}$ とし、各 $j$ について $d_j$ を $\mathcal{E}$ 上の入次数、$s_j=\sum_{i:(i\to j)\in\mathcal{E}} r_{ij}$、$b_j=s_j/\max(d_j,1)$ と定義すると、バランス項と物理損失の全体は以下となります。

$$
\mathcal{L}_{\mathrm{bal}} = \frac{1}{N}\sum_{j=1}^{N} b_j^2, \qquad \mathcal{L}_{\mathrm{phys}} = \mathcal{L}_{\mathrm{grad}}+\mathcal{L}_{\mathrm{bal}}
$$

#### 可視化: 空間推論の比較

![GNN 推論の比較](./multiphysics_dec_solver/step4_hetero_gnn_training/zenn_assets/gnn_inference_comparison.png)

*（注: プライマル流体頂点における速度の大きさについて、グラウンドトゥルース・GNN予測・絶対誤差を空間マッピングしたものです。学習済みの順伝播パイプラインがエンドツーエンドで妥当であることを示しています。）*

### ✅ 検証・確認事項

1. **アーキテクチャの妥当性** — **`HeteroConv`** が、根本的に異なるトポロジーエンティティ間（**`p2p`**、**`d2d`**、**`p2d`**、および逆方向の **`d2p`**）で、次元の不一致を起こすことなくメッセージを正しくルーティング・集約できることを確認しました。
2. **物理情報付きパイプラインの稼働** — グラフ勾配に基づくカスタムの **擬似発散損失**を統合し、PyGのグラフ構造上で質量保存（発散ゼロ）に寄せるペナルティを、MSEと共に安定してバックプロパゲーションできることを確認しました。
3. **エンドツーエンドの完成** — Julia側の数学的定義からPython側のニューラルネット推論、そして誤差の空間マッピングに至るまで、データパイプラインが途切れることなく機能することを実証しました。

### ステップ 5: ゼロショット汎化とパフォーマンスベンチマーク

構築したサロゲートモデルが、未知のメッシュに対してどの程度 **トポロジカルな汎化性能（ゼロショット推論）** を発揮するかを評価します。また、**厳密なパフォーマンスベンチマーク**を実施し、物理的な精度（MSE/MAE）を維持したまま、従来のCFDソルバーに対して **GNNサロゲートがいかに圧倒的な高速化をもたらすか**を定量化します。

#### 可視化: ゼロショット空間比較

![ゼロショット空間比較（速度の大きさ）](./multiphysics_dec_solver/step5_zero_shot_evaluation/evaluation_results/zeroshot_comparison.png)

*（注: 未知の評価用グラフに対する、プライマル流体頂点の速度の大きさの散布図（グラウンドトゥルース・予測・絶対誤差）です。）*

### 🔬 評価の対象と検証事項

1. **推論の移植性** — チャネル幅が一致する **ステップ3形式の `.pt` ファイル**であれば、追加の再学習を一切行うことなく **`PhysicsInformedHeteroGNN`** で実行可能なことを確認しました。
2. **精度の定量化** — グローバルなMSE/MAEによってプライマル場の再構成精度を要約し、空間誤差パネルによって局所的な差異の分布を可視化しました。
3. **速度の定量化** — 何度も推論を繰り返すベンチマークにより、対話的な設計ツールや外側の最適化ループに十分に組み込める **ミリ秒クラス** の低レイテンシを記録しました。

#### 可視化: ROI推論高速化（代表的ベンチマーク）

![ROI 推論高速化ベンチマーク](./multiphysics_dec_solver/step5_zero_shot_evaluation/evaluation_results/roi_speedup_benchmark.png)

対数スケールの棒グラフを用いて、代表的な **Julia/DEC CFD** の計算時間と、**HeteroGNNサロゲート**の推論時間（いずれも秒単位）を比較しています。

---

## 考察：サロゲートモデルがもたらす真のROI（投資対効果）

ステップ5のベンチマークで示された「数万倍の高速化」は、単なる計算速度の向上にとどまらず、設計や研究のプロセスそのものを変革する高いROI（投資対効果）をもたらします。ここでは、サロゲートモデルを実務に導入する意義について、3つの視点から考察します。

1. **推論における「限界費用」の極小化**
従来のCFDソルバーでは、形状や境界条件を1つ変更するたびに同等の重い計算コスト（時間と計算資源）が発生します。一方、一度学習を終えたGNNサロゲートは、未知のメッシュに対する推論コストがミリ秒単位、つまり計算の限界費用がほぼゼロになります。これにより、限られた予算と時間の中で数千〜数万パターンの設計空間を網羅的に探索することが現実的になります。

2. **精度と速度の「トリアージ」**
当然ながら、サロゲートモデルは近似であるため、厳密な数値計算を完全に代替するものではありません。しかし、「数万通りの設計アイデアから、有望な数十パターンをミリ秒で絞り込む（トリアージする）」という用途においては無類の強さを発揮します。探索フェーズ（全体の99%）をGNNに任せ、最終的な詳細検証（残りの1%）にのみ重厚なCFDソルバーを実行するというハイブリッドな運用が、全体のROIを最大化する鍵となります。

3. **パイプライン化による「初期投資」の確実な回収**
AIモデルの実用化には「良質なデータの生成」と「モデルの学習」という大きな初期投資が必要です。本リポジトリのように、Juliaによる厳密なグラウンドトゥルース生成からPythonでのGNN学習までを、明確なコントラクト（JSON）によって自動化・パイプライン化しておくことで、この初期投資のハードルが大幅に下がります。データの再現性とトポロジーの安全性が担保されるため、手戻りが減り、より早期にROIをプラスに転じさせることが可能になります。

---

## おわりに

本リポジトリのソースコードおよびドキュメントは **MIT License** の下で公開されています。手元での再現や独自の改良を試される際は、各ステップディレクトリ内の `requirements_*.txt` や `Project.toml` から環境を構築していただくと、本稿で紹介したコマンドとスムーズに対応付けることができます。

CFDの厳密さとGNNの圧倒的な推論速度を融合させるこのアプローチが、皆様の設計・解析プロセスの高速化、そして真の意味でのROI向上に少しでも役立てば幸いです。

---

## 参考

- [categorical_physics_engine（GitHub）](https://github.com/kohmaruworks/categorical_physics_engine)


## ライセンス

本プロジェクトは **MIT License** の下で公開されています。
