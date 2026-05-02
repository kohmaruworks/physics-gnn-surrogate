# Step 2 — Heterogeneous DEC graph export (JSON Contract V2).
#
# Reconstructs `EmbeddedDeltaDualComplex2D` from Step 1 topology (no edits to Step 1),
# extracts Primal/Dual/Hodge-related COO edge lists, and emits **0-based** indices for PyTorch.

using Catlab: nparts
using CombinatorialSpaces
using CombinatorialSpaces.SimplicialSets:
    EmbeddedDeltaSet2D,
    add_vertices!,
    glue_triangle!,
    orient!,
    edge_vertices,
    parts
using CombinatorialSpaces.DiscreteExteriorCalculus:
    EmbeddedDeltaDualComplex2D,
    elementary_duals,
    subdivide_duals!,
    Barycenter
using GeometryBasics: Point3
using JSON3
using JLD2

const Point3d = Point3{Float64}

"""Mirror of Step 1 `TopologyBlocks1Based` for `JLD2` deserialization (must stay aligned)."""
struct TopologyBlocks1Based
    num_vertices::Int
    vertex_xyz::Vector{Vector{Float64}}
    undirected_edges::Vector{Tuple{Int,Int}}
    triangles::Vector{Tuple{Int,Int,Int}}
    vertex_tags::Vector{String}
end

function build_primal_delta_set(topo::TopologyBlocks1Based)::EmbeddedDeltaSet2D
    s = EmbeddedDeltaSet2D{Bool,Point3d}()
    add_vertices!(s, topo.num_vertices)
    for v in 1:topo.num_vertices
        xyz = topo.vertex_xyz[v]
        z = length(xyz) >= 3 ? Float64(xyz[3]) : 0.0
        s[v, :point] = Point3d(Float64(xyz[1]), Float64(xyz[2]), z)
    end
    for (i, j, k) in topo.triangles
        glue_triangle!(s, Int(i), Int(j), Int(k))
    end
    orient!(s)
    return s
end

"""Rebuild dual complex exactly as Step 1 (`Barycenter` subdivision)."""
function build_dual_complex(topo::TopologyBlocks1Based)::EmbeddedDeltaDualComplex2D
    primal = build_primal_delta_set(topo)
    # Match Step 1 `mesh_gen.jl`: typed dual complex + barycentric subdivision.
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3d}(primal)
    subdivide_duals!(sd, Barycenter())
    return sd
end

"""Primal vertex adjacency (bidirected COO) — supports gradient / `d₀` style messaging."""
function primal_vertex_edge_coo(sd::EmbeddedDeltaDualComplex2D)
    src = Int[]
    dst = Int[]
    for e in parts(sd, :E)
        v0, v1 = edge_vertices(sd, e)
        push!(src, Int(v0), Int(v1))
        push!(dst, Int(v1), Int(v0))
    end
    return src, dst
end

"""Dual control-volume adjacency (bidirected COO on `DualV`)."""
function dual_vertex_edge_coo(sd::EmbeddedDeltaDualComplex2D)
    src = Int[]
    dst = Int[]
    for de in parts(sd, :DualE)
        a = Int(sd[de, :D_∂v0])
        b = Int(sd[de, :D_∂v1])
        push!(src, a, b)
        push!(dst, b, a)
    end
    return src, dst
end

"""
Primal edge → dual edge incidence used by DEC/Hodge-1 pipelines (`elementary_duals` on primal edges).
COO rows: `(primal_edge_idx, dual_edge_idx)` in **1-based** Julia indexing before export.
"""
function primal_edge_to_dual_edge_coo(sd::EmbeddedDeltaDualComplex2D)
    src = Int[]
    dst = Int[]
    for e in parts(sd, :E)
        for de in elementary_duals(Val(1), sd, Int(e))
            push!(src, Int(e))
            push!(dst, Int(de))
        end
    end
    return src, dst
end

function subtract_one!(xs::Vector{Int})
    xs .= xs .- 1
end

function assert_vertex_edges(sb::AbstractVector{Int}, db::AbstractVector{Int}, nv::Int, msg::String)
    isempty(sb) && return
    @assert all(>=(0), sb) && all(>=(0), db) "Negative index after 0-based remap ($msg)"
    @assert maximum(sb) < nv && maximum(db) < nv "Edge index out of range for $nv nodes ($msg)"
end

function assert_primal_edges_to_dual_edges(ps::AbstractVector{Int}, ds::AbstractVector{Int}, ne::Int, nde::Int)
    isempty(ps) && return
    @assert all(>=(0), ps) && all(>=(0), ds)
    @assert maximum(ps) < ne && maximum(ds) < nde "Primal-to-dual COO out of range (ne=$ne, nde=$nde)"
end

function primal_xy_coordinates(sd::EmbeddedDeltaDualComplex2D)
    n = nparts(sd, :V)
    [[Float64(sd[v, :point][1]), Float64(sd[v, :point][2])] for v in 1:n]
end

function dual_xy_coordinates(sd::EmbeddedDeltaDualComplex2D)
    n = nparts(sd, :DualV)
    [[Float64(sd[dv, :dual_point][1]), Float64(sd[dv, :dual_point][2])] for dv in 1:n]
end

function load_step1_jld2(path::AbstractString)
    jldopen(path, "r") do f
        times = Vector{Float64}(read(f, "time"))
        vx = Matrix{Float64}(read(f, "velocity_vertex_vx"))
        vy = Matrix{Float64}(read(f, "velocity_vertex_vy"))
        pr = Matrix{Float64}(read(f, "pressure"))
        topo = read(f, "topology")::TopologyBlocks1Based
        return (; times, vx, vy, pr, topo)
    end
end

"""Rebuild `TopologyBlocks1Based` from Step 1 JSON `topology_payload` (flat 1-based incidence)."""
function topology_from_step1_json_payload(topo)::TopologyBlocks1Based
    xyz = Vector{Vector{Float64}}(collect(topo.vertex_xyz))
    tags = Vector{String}(collect(topo.vertex_tags))
    n = length(xyz)
    flat_e = Vector{Int}(collect(topo.undirected_edges_flat_1based))
    flat_tris = Vector{Int}(collect(topo.triangles_flat_1based))
    edges = Tuple{Int,Int}[]
    for i in 1:2:(length(flat_e)-1)
        push!(edges, (flat_e[i], flat_e[i+1]))
    end
    tris = Tuple{Int,Int,Int}[]
    for i in 1:3:(length(flat_tris)-2)
        push!(tris, (flat_tris[i], flat_tris[i+1], flat_tris[i+2]))
    end
    TopologyBlocks1Based(n, xyz, edges, tris, tags)
end

function load_step1_json(path::AbstractString)
    raw = JSON3.read(read(path, String))
    times = Vector{Float64}(collect(raw.time))
    frames = raw.node_features_time_series
    np = length(times)
    nv = length(frames[1])
    vx = zeros(Float64, np, nv)
    vy = zeros(Float64, np, nv)
    pr = zeros(Float64, np, nv)
    for it in 1:np
        fr = frames[it]
        @assert length(fr) == nv "Frame $it length mismatch"
        for j in 1:nv
            vx[it, j] = Float64(fr[j][1])
            vy[it, j] = Float64(fr[j][2])
            pr[it, j] = Float64(fr[j][3])
        end
    end
    topo = topology_from_step1_json_payload(raw.topology)
    return (; times, vx, vy, pr, topo)
end

function load_step1_ground_truth(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    if ext == ".jld2"
        return load_step1_jld2(path)
    elseif ext == ".json"
        return load_step1_json(path)
    else
        throw(ArgumentError("Unsupported Step 1 format (use .jld2 or .json): $(path)"))
    end
end

function nearest_time_index(times::AbstractVector{<:Real}, t_target::Float64)::Int
    return argmin(i -> abs(Float64(times[i]) - t_target), eachindex(times))
end

"""
Build Contract V2 dict from Step 1 topology + primal vertex fields at chosen time row `it`.
"""
function hetero_contract_payload(
    topo::TopologyBlocks1Based,
    velocity_u::AbstractVector{Float64},
    velocity_v::AbstractVector{Float64},
    pressure::AbstractVector{Float64};
    time_value::Float64,
    source_path::AbstractString,
)::Dict{String,Any}
    length(velocity_u) == topo.num_vertices || throw(ArgumentError("velocity_u length mismatch"))
    length(velocity_v) == topo.num_vertices || throw(ArgumentError("velocity_v length mismatch"))
    length(pressure) == topo.num_vertices || throw(ArgumentError("pressure length mismatch"))

    sd = build_dual_complex(topo)
    nv = nparts(sd, :V)
    ndv = nparts(sd, :DualV)
    ne = nparts(sd, :E)
    nde = nparts(sd, :DualE)
    @assert nv == topo.num_vertices "Reconstructed primal vertex count differs from topology"

    pp_s, pp_d = primal_vertex_edge_coo(sd)
    dd_s, dd_d = dual_vertex_edge_coo(sd)
    pd_s, pd_d = primal_edge_to_dual_edge_coo(sd)

    subtract_one!(pp_s)
    subtract_one!(pp_d)
    subtract_one!(dd_s)
    subtract_one!(dd_d)
    subtract_one!(pd_s)
    subtract_one!(pd_d)

    assert_vertex_edges(pp_s, pp_d, nv, "primal_to_primal")
    assert_vertex_edges(dd_s, dd_d, ndv, "dual_to_dual")
    assert_primal_edges_to_dual_edges(pd_s, pd_d, ne, nde)

    pcoords = primal_xy_coordinates(sd)
    dcoords = dual_xy_coordinates(sd)
    @assert length(pcoords) == nv && length(dcoords) == ndv

    return Dict{String,Any}(
        "schema" => "categorical_physics_engine_step2_v2",
        "indexing" => "All edge_index entries are **0-based** for PyTorch / PyG.",
        "time" => time_value,
        "step1_source" => source_path,
        "nodes" => Dict{String,Any}(
            "primal" => Dict{String,Any}(
                "num_nodes" => nv,
                "coordinates" => pcoords,
            ),
            "dual" => Dict{String,Any}(
                "num_nodes" => ndv,
                "coordinates" => dcoords,
            ),
        ),
        "edges" => Dict{String,Any}(
            "primal_to_primal" => Dict{String,Any}("edge_index" => [pp_s, pp_d]),
            "dual_to_dual" => Dict{String,Any}("edge_index" => [dd_s, dd_d]),
            "primal_to_dual" => Dict{String,Any}("edge_index" => [pd_s, pd_d]),
        ),
        "features" => Dict{String,Any}(
            "velocity_u" => Vector{Float64}(velocity_u),
            "velocity_v" => Vector{Float64}(velocity_v),
            "pressure" => Vector{Float64}(pressure),
        ),
        "dec_counts" => Dict{String,Int}(
            "num_primal_vertices" => nv,
            "num_dual_vertices" => ndv,
            "num_primal_edges" => ne,
            "num_dual_edges" => nde,
            "num_primal_triangles" => nparts(sd, :Tri),
            "num_dual_triangles" => nparts(sd, :DualTri),
        ),
    )
end

"""
Export JSON Contract V2 from Step 1 `ground_truth_*.jld2`.

`topology` must deserialize as `TopologyBlocks1Based` (mirrored struct above).
"""
function export_hetero_json_v2(
    step1_input_path::AbstractString,
    out_json_path::AbstractString;
    t_target::Float64 = 0.35,
)
    isfile(step1_input_path) || error("Step 1 ground-truth file not found: $(step1_input_path)")
    data = load_step1_ground_truth(step1_input_path)
    it = nearest_time_index(data.times, t_target)
    t_sel = Float64(data.times[it])
    vu = Vector{Float64}(data.vx[it, :])
    vv = Vector{Float64}(data.vy[it, :])
    pr = Vector{Float64}(data.pr[it, :])

    payload = hetero_contract_payload(data.topo, vu, vv, pr; time_value = t_sel, source_path = string(step1_input_path))

    mkpath(dirname(out_json_path))
    open(out_json_path, "w") do io
        JSON3.pretty(io, payload)
    end
    @info "Wrote JSON Contract V2" out_json_path time_selected = t_sel time_index = it
    return payload
end
