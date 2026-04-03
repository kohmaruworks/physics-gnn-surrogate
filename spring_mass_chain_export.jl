# 5 質点の 1 次元直列バネ–質量系を Catlab の有向グラフとして表し、
# 初期位置・初速度を x、DifferentialEquations.jl で求めた時刻 t1 の状態を y として JSON エクスポートする。
#
# 物理モデル: 各質点 m、隣接質点間のバネ k（フックの法則）。自由端（壁なし）。
#   u1'' = (k/m)(u2-u1),  ui'' = (k/m)(u_{i+1}-2ui+u_{i-1}),  un'' = (k/m)(u_{n-1}-un)
#
# 実行: julia --project=. spring_mass_chain_export.jl
#
# グラフ: 頂点 i と i+1 の間にバネ → 隣接ペアを双方向の有向辺で表す（相互作用の対称性）。

using DifferentialEquations
using Random

include(joinpath(@__DIR__, "export_catlab_graph_json.jl"))

"""質点数 `n` の直列チェーン。辺は (i,i+1) と (i+1,i) の両方向。"""
function spring_mass_chain_graph(n::Int)
    n ≥ 2 || throw(ArgumentError("n must be at least 2, got $n"))
    g = Graph()
    add_vertices!(g, n)
    src = Int[]
    tgt = Int[]
    for i in 1:(n - 1)
        push!(src, i, i + 1)
        push!(tgt, i + 1, i)
    end
    add_edges!(g, src, tgt)
    g
end

"""
連立 1 階化した状態 `z = [u_1,…,u_n, v_1,…,v_n]`（位置・速度）の時間発展。
`p = (n, k, m)`。
"""
function spring_mass_chain_ode!(dz::AbstractVector, z::AbstractVector, p, _t)
    n, k, m = p
    u = @view z[1:n]
    v = @view z[n+1:2n]
    @views dz[1:n] .= v
    a = @view dz[n+1:2n]
    km = k / m
    a[1] = km * (u[2] - u[1])
    for i in 2:(n - 1)
        a[i] = km * (u[i + 1] - 2 * u[i] + u[i - 1])
    end
    a[n] = km * (u[n - 1] - u[n])
    nothing
end

"""
`x0`: 初期条件行列 (n×2) — 列1=位置、列2=速度。
`tspan = (0, t1)` を積分し、時刻 `t1` の位置・速度を (n×2) の `y` として返す。
"""
function integrate_spring_mass_chain(x0::AbstractMatrix, n::Int, k::Float64, m::Float64, t1::Float64)
    size(x0, 1) == n && size(x0, 2) == 2 ||
        throw(ArgumentError("x0 must be n×2, got $(size(x0,1))×$(size(x0,2)) with n=$n"))
    z0 = vcat(x0[:, 1], x0[:, 2])
    p = (n, k, m)
    prob = ODEProblem(spring_mass_chain_ode!, z0, (0.0, t1), p)
    sol = solve(prob, Tsit5(); saveat = t1, abstol = 1e-10, reltol = 1e-10)
    isempty(sol.u) && error("ODE solution empty")
    z1 = sol.u[end]
    hcat(z1[1:n], z1[n+1:2n])
end

function main(;
    n::Int = 5,
    seed::Union{Nothing,Integer} = 42,
    outfile = "spring_mass_chain_5.json",
    k::Float64 = 1.0,
    m::Float64 = 1.0,
    t1::Float64 = 0.1,
)
    isnothing(seed) || Random.seed!(seed)

    g = spring_mass_chain_graph(n)
    x = randn(Float64, n, 2)
    y = integrate_spring_mass_chain(x, n, k, m, t1)

    path = joinpath(@__DIR__, outfile)
    save_catlab_graph_json(path, g; x=x, y=y, pretty=true)
    println("exported: ", path)
    println("  nv = ", nv(g), ", ne = ", ne(g))
    println("  x: initial position & velocity at t=0")
    println("  y: position & velocity at t=$(t1) (from DifferentialEquations.jl, k=$k, m=$m)")
    path
end

main()
