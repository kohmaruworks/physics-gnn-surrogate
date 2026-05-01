# メッシュ + 多物理 decapode 雛形 + 参照系時系列（toy: グラフ拡散 on T, 凍結 vx,vy,p）+ `data/interim` JSON。
# 依存: OrdinaryDiffEq, 01, 02, utils_export
#
# 本番: Decapodes `generate` で dual complex 上の ODE 関数に接続。ここはパイプラインと JSON スキーマ検証用。

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq

const _ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const _JUL = @__DIR__

include(joinpath(_JUL, "01_mesh_generation.jl"))
include(joinpath(_JUL, "02_physics_modeling.jl"))
include(joinpath(_JUL, "utils_export.jl"))

function graph_laplacian(n::Int, undirected::Vector{Tuple{Int,Int}})
    A = spzeros(Float64, n, n)
    for (a, b) in undirected
        A[a, b] = 1.0
        A[b, a] = 1.0
    end
    deg = [sum(@view A[i, :]) for i in 1:n]
    spdiagm(0 => deg) - A
end

struct ToyODEParams
    n::Int
    L::AbstractMatrix{Float64}
    α_heat::Float64
end

function toy_semi_discrete_heat!(du, u, p::ToyODEParams, _t)
    fill!(du, 0.0)
    Tvec = [u[4i] for i in 1:p.n]
    dT = p.L * Tvec
    for i in 1:p.n
        du[4i] = -p.α_heat * dT[i]
    end
    return nothing
end

function feature_matrix_nx4(u::AbstractVector, n::Int)
    hcat(
        [u[4(i - 1) + 1] for i in 1:n],
        [u[4(i - 1) + 2] for i in 1:n],
        [u[4(i - 1) + 3] for i in 1:n],
        [u[4(i - 1) + 4] for i in 1:n],
    )
end

function default_interim_path()
    joinpath(_ROOT, "data", "interim", "v2_step1_ground_truth_toy.json")
end

const INTERIM_SCHEMA_V2 = "physics_gnn_interim_v2"

"""各 Primal 三角形セル上で、種々の時間ステップにおける頂点特徴のバリセンター平均（dual ノード = 三角形 1 本対 1）。"""
function dual_features_triangle_avg(F::AbstractMatrix{<:Real}, mesh)
    primal = mesh.primal
    nT = mesh.n_triangles
    nf = size(F, 2)
    out = zeros(Float64, nT, nf)
    for t in 1:nT
        i, j, k = map(Int, triangle_vertices(primal, t))
        out[t, :] .= (collect(F[i, :]) .+ collect(F[j, :]) .+ collect(F[k, :])) ./ 3
    end
    out
end

function run_ground_truth_pipeline(;
    tspan = (0.0, 0.5),
    saveat = 0.05,
    α_heat = 0.1,
    outfile = default_interim_path(),
)
    mesh = build_primal_dual_mesh()
    n = mesh.n_vertices
    Lm = graph_laplacian(n, collect(mesh.undirected_edges_1based))
    p = ToyODEParams(n, Lm, α_heat)
    u0 = zeros(4n)
    for i in 1:n
        u0[4(i - 1) + 1] = 0.1
        u0[4(i - 1) + 2] = 0.0
        u0[4(i - 1) + 3] = 0.0
        u0[4(i - 1) + 4] = 0.2 * sin(i)
    end

    model = build_coupled_multiphysics_model()
    _ = model.coupled_fluid_heat
    _ = model.operadic_compose_demo

    prob = ODEProblem(toy_semi_discrete_heat!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1e-8, reltol = 1e-6)
    tvec = Array(sol.t)
    feature_series = [feature_matrix_nx4(uk, n) for uk in sol.u]

    top = topology_dict_v2(;
        primal_edges_1based = collect(mesh.undirected_edges_1based),
        dual_edges_1based = collect(mesh.dual_triangle_adjacency_1based),
        primal_to_dual_pairs_1based = collect(mesh.primal_vertex_triangle_incidence_1based),
        triangles_1based = collect(mesh.triangles_1based),
    )

    dual_mat0 = dual_features_triangle_avg(feature_series[1], mesh)
    dual_feature_rows = [collect(@view dual_mat0[t, :]) for t in axes(dual_mat0, 1)]
    dual_ts = [begin
            dm = dual_features_triangle_avg(F, mesh)
            [collect(@view dm[t, :]) for t in axes(dm, 1)]
        end for F in feature_series]

    payload = Dict{String,Any}(
        "schema" => INTERIM_SCHEMA_V2,
        "indexing_note" => string(
            "All topology indices are 0-based for PyTorch Geometric. ",
            "Julia 1-based indices are converted exactly once via utils_export. ",
            "dual_node i (0-based) identifies primal triangle i (PyG dual_node ⟷ primal triangle cell).",
        ),
        "time" => tvec,
        "num_nodes" => n,
        "num_dual_nodes" => mesh.n_dual_nodes,
        "node_feature_names" => ["vx", "vy", "p", "T"],
        "node_features_time_series" => [collect(eachrow(F)) for F in feature_series],
        "dual_node_features" => dual_feature_rows,
        "dual_node_features_time_series" => dual_ts,
        "topology" => top,
    )
    save_interim_json(payload, outfile; pretty = true)
    (solution = sol, path = outfile, payload = payload, mesh = mesh, model = model)
end

if abspath(PROGRAM_FILE) == @__FILE__
    r = run_ground_truth_pipeline()
    println("Wrote: ", r.path)
end
