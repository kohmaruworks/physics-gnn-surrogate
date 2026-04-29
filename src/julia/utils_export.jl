# JSON3 と 1-based → 0-based 単一関所。
# 依存: JSON3

using JSON3

function convert_to_0based_index(x::Integer)
    Int(x) - 1
end

function convert_to_0based_index(a::AbstractVector{<:Integer})
    [convert_to_0based_index(v) for v in a]
end

function convert_to_0based_index(m::AbstractMatrix{<:Integer})
    convert_to_0based_index.(m)
end

function convert_to_0based_index(t::Tuple{Vararg{Integer}})
    Tuple(convert_to_0based_index.(t))
end

function convert_to_0based_index(t::NTuple{3,T}) where {T<:Integer}
    (Int(t[1]) - 1, Int(t[2]) - 1, Int(t[3]) - 1)
end

function convert_to_0based_index(t::NTuple{2,T}) where {T<:Integer}
    (Int(t[1]) - 1, Int(t[2]) - 1)
end

"""PyG 向け `edge_index` 形状 (2, E), **0-based**（入力は 1-based の無向辺リスト）。"""
function edge_index_0based_from_undirected(
    edges::Vector{Tuple{Int,Int}};
    bidirectional::Bool = true,
)
    src = Int[]
    tgt = Int[]
    for (a, b) in edges
        sa = convert_to_0based_index(a)
        sb = convert_to_0based_index(b)
        if bidirectional
            push!(src, sa, sb)
            push!(tgt, sb, sa)
        else
            push!(src, sa)
            push!(tgt, sb)
        end
    end
    L = length(src)
    L == 0 && return Matrix{Int}(undef, 2, 0)
    m = Matrix{Int}(undef, 2, L)
    m[1, :] .= src
    m[2, :] .= tgt
    m
end

function triangles_0based(tris::Vector{Tuple{Int,Int,Int}})
    isempty(tris) && return Matrix{Int}(undef, 3, 0)
    o = Matrix{Int}(undef, 3, length(tris))
    for (k, t) in enumerate(tris)
        c = convert_to_0based_index(t)
        o[1, k], o[2, k], o[3, k] = c[1], c[2], c[3]
    end
    o
end

"""PyG / JSON 用: `edge_index` を行優先フラット [src..., tgt...]（`dataset._as_long_edge_index` と整合）。"""
function edge_index_flat_rowmajor(m::AbstractMatrix{<:Integer})
    size(m, 1) == 2 || throw(ArgumentError("edge_index は 2×E である必要があります"))
    E = size(m, 2)
    E == 0 && return Int[]
    vcat(Int.(collect(@view m[1, :])), Int.(collect(@view m[2, :])))
end

"""三角形行列 (3×F) を JSON 用 1 列ベクトル [i0,j0,k0, ...] に変換（0-based のまま値は保持）。"""
function triangles_flat_rowmajor(m::AbstractMatrix{<:Integer})
    size(m, 1) == 3 || throw(ArgumentError("triangles は 3×F である必要があります"))
    F = size(m, 2)
    F == 0 && return Int[]
    reduce(vcat, [Int.(collect(@view m[:, k])) for k in 1:F])
end

"""Phase 2 interim JSON Contract V2 用の `topology` 辞書（値は 0-based フラット）。"""
function topology_dict_v2(;
    primal_edges_1based::Vector{Tuple{Int,Int}},
    dual_edges_1based::Vector{Tuple{Int,Int}},
    primal_to_dual_pairs_1based::Vector{Tuple{Int,Int}},
    triangles_1based::Vector{Tuple{Int,Int,Int}},
    primal_bidirectional::Bool = true,
    dual_bidirectional::Bool = true,
)
    ep = edge_index_0based_from_undirected(primal_edges_1based; bidirectional = primal_bidirectional)
    ed = edge_index_0based_from_undirected(dual_edges_1based; bidirectional = dual_bidirectional)
    epd = edge_index_0based_from_undirected(primal_to_dual_pairs_1based; bidirectional = false)
    tri0 = triangles_0based(triangles_1based)
    Dict{String,Any}(
        "edge_index" => edge_index_flat_rowmajor(ep),
        "dual_edge_index" => edge_index_flat_rowmajor(ed),
        "primal_to_dual_edge_index" => edge_index_flat_rowmajor(epd),
        "triangles" => triangles_flat_rowmajor(tri0),
    )
end

function save_interim_json(data::AbstractDict{String,Any}, path::AbstractString; pretty::Bool = true)
    mkpath(dirname(path))
    open(path, "w") do io
        if pretty
            JSON3.pretty(io, data)
        else
            JSON3.write(io, data)
        end
    end
    path
end
