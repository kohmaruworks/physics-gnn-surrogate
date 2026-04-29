# DEC 用 Primal / Dual 2D 三角形メッシュ。
# 依存: CombinatorialSpaces, GeometryBasics
#
# 実行例: `julia --project=src/julia -e 'include("src/julia/01_mesh_generation.jl")'`

using CombinatorialSpaces
using CombinatorialSpaces: EmbeddedDeltaDualComplex2D, EmbeddedDeltaSet2D
# 現行 CombinatorialSpaces: DEC ではなく DiscreteExteriorCalculus（ドキュメント準拠）
using CombinatorialSpaces.DiscreteExteriorCalculus: Barycenter, subdivide_duals!
using CombinatorialSpaces.SimplicialSets: add_vertices!, glue_triangle!, orient!, triangle_vertices,
    triangle_edges, triangles
using GeometryBasics: Point3
using Catlab: nparts

const Point3d = Point3{Float64}

"""
単位正方形を 2 三角形に分割した非構造メッシュ（z=0）。チャネル／ヒートシンク試算の最小例。
"""
function build_unit_square_two_triangles_primal()::EmbeddedDeltaSet2D{Bool,Point3d}
    s = EmbeddedDeltaSet2D{Bool,Point3d}()
    add_vertices!(
        s,
        4;
        point = [
            Point3d(0.0, 0.0, 0.0),
            Point3d(1.0, 0.0, 0.0),
            Point3d(1.0, 1.0, 0.0),
            Point3d(0.0, 1.0, 0.0),
        ],
    )
    glue_triangle!(s, 1, 2, 3; tri_orientation = true)
    glue_triangle!(s, 1, 3, 4; tri_orientation = true)
    s[:edge_orientation] = true
    orient!(s)
    s
end

function build_dual_complex(
    primal::EmbeddedDeltaSet2D{Bool,Point3d},
)::EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}
    dual = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(primal)
    subdivide_duals!(dual, Barycenter())
    dual
end

function collect_undirected_edges_1based(primal::EmbeddedDeltaSet2D)
    seen = Set{Tuple{Int,Int}}()
    nT = nparts(primal, :Tri)
    nT > 0 || return Tuple{Int,Int}[]
    for t in 1:nT
        i, j, k = map(Int, triangle_vertices(primal, t))
        for (a, b) in ((i, j), (j, k), (k, i))
            e = a < b ? (a, b) : (b, a)
            push!(seen, e)
        end
    end
    collect(seen)
end

function collect_triangles_1based(primal::EmbeddedDeltaSet2D)
    nT = nparts(primal, :Tri)
    map(1:nT) do t
        i, j, k = map(Int, triangle_vertices(primal, t))
        (i, j, k)
    end
end

"""隣接する三角形ペア（各内部稜に対し 1 本）。DEC の双対 1-複体上で三角形重心（`tri_center`）同士が稜で結ばれる関係と一致する。"""
function collect_dual_triangle_adjacency_1based(primal::EmbeddedDeltaSet2D)
    edge_tris = Dict{Int,Vector{Int}}()
    for t in triangles(primal)
        for e in triangle_edges(primal, t)
            push!(get!(Vector{Int}, edge_tris, e), Int(t))
        end
    end
    seen = Set{Tuple{Int,Int}}()
    for ts in values(edge_tris)
        length(ts) == 2 || continue
        t1, t2 = sort(ts)
        push!(seen, (t1, t2))
    end
    collect(seen)
end

"""Primal 頂点と Primal 三角形（= PyG 側の dual ノードインデックス）の包含関係。各角 1 本の有向ペア (vertex, triangle)。"""
function collect_primal_vertex_triangle_incidence_1based(primal::EmbeddedDeltaSet2D)
    pairs = Tuple{Int,Int}[]
    for t in triangles(primal)
        i, j, k = map(Int, triangle_vertices(primal, t))
        for v in (i, j, k)
            push!(pairs, (v, Int(t)))
        end
    end
    pairs
end

"""
`build_primal_dual_mesh` の返り値。Primal・Dual、および JSON 用の 1-based 位相補助データ。
"""
function build_primal_dual_mesh(; scale::Real = 1.0)
    primal = build_unit_square_two_triangles_primal()
    if !isone(scale)
        for v in 1:nparts(primal, :V)
            p = primal[v, :point]
            primal[v, :point] = Point3d(scale * p[1], scale * p[2], scale * p[3])
        end
    end
    dual = build_dual_complex(primal)
    edges = collect_undirected_edges_1based(primal)
    tris = collect_triangles_1based(primal)
    nT = nparts(primal, :Tri)
    dual_tri_adj = collect_dual_triangle_adjacency_1based(primal)
    v_to_t = collect_primal_vertex_triangle_incidence_1based(primal)
    return (
        primal = primal,
        dual = dual,
        n_vertices = nparts(primal, :V),
        n_edges = length(edges),
        n_triangles = nT,
        n_dual_nodes = nT,
        undirected_edges_1based = edges,
        triangles_1based = tris,
        dual_triangle_adjacency_1based = dual_tri_adj,
        primal_vertex_triangle_incidence_1based = v_to_t,
    )
end
