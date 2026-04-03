# Catlab.Graphs の有向グラフ（Graph）を PyTorch Geometric 向け JSON に書き出す。
#
# 初回のみ: julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# 頂点番号は JSON 内で 0 始まり（PyG の edge_index と一致）。

using Catlab
using Catlab.Graphs
using JSON3

"""
`g` の辺集合を (src, tgt) のペアにし、頂点 ID は 1-based（Julia）から 0-based に変換する。
"""
function catlab_graph_edges_0based(g)
    pairs = Vector{Vector{Int}}()
    for e in edges(g)
        push!(pairs, [src(g, e) - 1, tgt(g, e) - 1])
    end
    pairs
end

function _feature_matrix_to_json(A::AbstractMatrix)
    [collect(@view A[i, :]) for i in axes(A, 1)]
end

"""
有向グラフ `g` を辞書に変換する。
- `num_nodes`: `nv(g)`（孤立頂点も含む）
- `edges`: `[[s,t], ...]`（0-based）
- `x`: 省略可。`Matrix` または `Vector{Vector}` で (num_nodes, feat_dim)
- `y`: 省略可。ノードごとのターゲット特徴（形状は `x` と同じ想定）
"""
function catlab_graph_to_dict(g; x=nothing, y=nothing)
    d = Dict{String,Any}(
        "format" => "catlab_directed_graph_v1",
        "num_nodes" => nv(g),
        "edges" => catlab_graph_edges_0based(g),
    )
    if !isnothing(x)
        d["x"] = x isa AbstractMatrix ? _feature_matrix_to_json(x) : x
    end
    if !isnothing(y)
        d["y"] = y isa AbstractMatrix ? _feature_matrix_to_json(y) : y
    end
    d
end

function save_catlab_graph_json(path::AbstractString, g; x=nothing, y=nothing, pretty::Bool=false)
    d = catlab_graph_to_dict(g; x=x, y=y)
    open(path, "w") do io
        if pretty
            JSON3.pretty(io, d)
        else
            JSON3.write(io, d)
        end
    end
    path
end

# ---- 使用例（`include` した場合はこのブロックをコメントアウトしてもよい） ----
if abspath(PROGRAM_FILE) == @__FILE__
    g = Graph()
    add_vertices!(g, 3)
    add_edges!(g, [1, 2], [2, 3])
    # 例: 頂点特徴 (3 頂点, 次元 2)
    x = Float64[1 0; 0 1; 1 1]
    out = joinpath(@__DIR__, "graph_from_catlab.json")
    save_catlab_graph_json(out, g; x=x, pretty=true)
    println("wrote ", out)
end
