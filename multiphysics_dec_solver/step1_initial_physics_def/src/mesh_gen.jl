# Primal simplicial mesh + `EmbeddedDeltaDualComplex2D` (Barycenter duals).
# All incidence exported to JSON lists Julia **1-based** indices; Python loaders subtract one once at ingest.

using CombinatorialSpaces
using CombinatorialSpaces: EmbeddedDeltaDualComplex2D, EmbeddedDeltaSet2D
using CombinatorialSpaces.DiscreteExteriorCalculus: Barycenter, subdivide_duals!
using CombinatorialSpaces.SimplicialSets: orient!, triangle_vertices, triangles
using GeometryBasics: Point3
using Catlab: nparts

const Point3d = Point3{Float64}

function collect_undirected_edges_1based(primal::EmbeddedDeltaSet2D)::Vector{Tuple{Int,Int}}
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

function collect_triangles_1based(primal::EmbeddedDeltaSet2D)::Vector{Tuple{Int,Int,Int}}
    nT = nparts(primal, :Tri)
    map(1:nT) do t
        i, j, k = map(Int, triangle_vertices(primal, t))
        (i, j, k)
    end
end

struct TopologyBlocks1Based
    num_vertices::Int
    vertex_xyz::Vector{Vector{Float64}}
    undirected_edges::Vector{Tuple{Int,Int}}
    triangles::Vector{Tuple{Int,Int,Int}}
    vertex_tags::Vector{String}
end

function classify_vertex_tag(
    xy::Tuple{Float64,Float64},
    scenario::Symbol,
    lx::Float64,
    ly::Float64,
    cyl_center::Tuple{Float64,Float64},
    cyl_radius::Float64,
)::String
    x, y = xy
    atol = 1e-9 * max(lx, ly)
    if scenario === :cylinder_wake
        if (x - cyl_center[1])^2 + (y - cyl_center[2])^2 <= cyl_radius^2 + 1e-11
            return "cylinder_wall"
        elseif y <= atol
            return "channel_bottom"
        elseif y >= ly - atol
            return "channel_top"
        elseif x <= atol
            return "inflow"
        elseif x >= lx - atol
            return "outflow"
        else
            return "fluid"
        end
    elseif scenario === :heat_sink
        if y <= atol
            return "cold_plate"
        elseif y >= ly - atol
            return "ambient"
        elseif x <= atol || x >= lx - atol
            return "side_wall"
        else
            return "sink_interior"
        end
    else
        throw(ArgumentError("Unknown scenario $(scenario); use :cylinder_wake or :heat_sink"))
    end
end

"""Structured triangle strip approximating channel flow / thermal slug scenarios."""
function build_scenario_mesh(scenario::Symbol; lx::Float64 = 2.0, ly::Float64 = 1.0, nx::Int = 40, ny::Int = 20)
    nx >= 2 && ny >= 2 || throw(ArgumentError("nx and ny must be ≥ 2"))
    rect = triangulated_grid(lx, ly; nx = nx, ny = ny, point_type = Point3d)
    orient!(rect)

    cyl_center = (0.35lx, 0.50ly)
    cyl_radius = min(lx, ly) * 0.09

    nV = nparts(rect, :V)
    verts_xyz = Vector{Vector{Float64}}(undef, nV)
    tags = Vector{String}(undef, nV)
    for v in 1:nV
        pt = rect[v, :point]
        verts_xyz[v] = [Float64(pt[1]), Float64(pt[2]), Float64(pt[3])]
        tags[v] = classify_vertex_tag((Float64(pt[1]), Float64(pt[2])), scenario, lx, ly, cyl_center, cyl_radius)
    end

    topo = TopologyBlocks1Based(
        nV,
        verts_xyz,
        collect_undirected_edges_1based(rect),
        collect_triangles_1based(rect),
        tags,
    )

    dual = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(rect)
    subdivide_duals!(dual, Barycenter())

    (; primal = rect, dual = dual, topology = topo, lx = lx, ly = ly, cyl_center = cyl_center, cyl_radius = cyl_radius)
end

"""Analytic fields used for initialization / auxiliary diagnostics."""
function prescribed_velocity_pressure(xyz::AbstractVector{<:Real}, scenario::Symbol)::Tuple{Float64,Float64,Float64}
    x, y = Float64(xyz[1]), Float64(xyz[2])
    if scenario === :cylinder_wake
        vx = 1.0 - 0.35 * exp(-((x - 0.65)^2 + (y - 0.5)^2) / 0.02)
        vy = 0.08 * sin(π * y)
    elseif scenario === :heat_sink
        vx = 0.05 * sin(π * x)
        vy = 0.35 * (1.0 - exp(-(y^2) / 0.08))
    else
        vx, vy = 0.0, 0.0
    end
    p = 1.0 - 0.5 * (vx^2 + vy^2)
    return vx, vy, p
end
