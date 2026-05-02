# Reference integration (`OrdinaryDiffEq`) + structured export for JSON/JLD2 interchange.

using OrdinaryDiffEq
using ComponentArrays
using Decapodes
using JSON3
using JLD2
using LinearAlgebra
using CombinatorialSpaces
using CombinatorialSpaces.SimplicialSets: nv, ne
using CombinatorialSpaces.DiscreteExteriorCalculus: ♭, DualVectorField, GeometricHodge, triangle_center

include(joinpath(@__DIR__, "definitions.jl"))
include(joinpath(@__DIR__, "mesh_gen.jl"))

"""Crude triangle-network averaging from primal `Form1` coefficients to vertex `(vx, vy)` proxies."""
function vertex_velocity_proxy(primal, V₁::AbstractVector)
    nvv = nv(primal)
    vx = zeros(Float64, nvv)
    vy = zeros(Float64, nvv)
    wsum = zeros(Float64, nvv)
    for e in 1:ne(primal)
        i = Int(primal[e, :∂v0])
        j = Int(primal[e, :∂v1])
        pi = primal[i, :point]
        pj = primal[j, :point]
        τx = Float64(pj[1] - pi[1])
        τy = Float64(pj[2] - pi[2])
        τz = Float64(pj[3] - pi[3])
        L = sqrt(τx^2 + τy^2 + τz^2)
        L < 1e-14 && continue
        txy = (τx / L, τy / L)
        ev = V₁[e]
        for v in (i, j)
            vx[v] += ev * txy[1]
            vy[v] += ev * txy[2]
            wsum[v] += 1.0
        end
    end
    for v in 1:nvv
        if wsum[v] > 0
            vx[v] /= wsum[v]
            vy[v] /= wsum[v]
        end
    end
    return vx, vy
end

function topology_payload(topo::TopologyBlocks1Based)
    flat_e = Int[]
    for (a, b) in topo.undirected_edges
        push!(flat_e, a, b)
    end
    flat_tris = Int[]
    for (i, j, k) in topo.triangles
        push!(flat_tris, i, j, k)
    end
    Dict{String,Any}(
        "undirected_edges_flat_1based" => flat_e,
        "triangles_flat_1based" => flat_tris,
        "vertex_tags" => topo.vertex_tags,
        "vertex_xyz" => topo.vertex_xyz,
    )
end

function initial_conditions(mesh_bundle, scenario::Symbol)
    sd = mesh_bundle.dual
    primal = mesh_bundle.primal

    vf(pt) = begin
        vx, vy, _ = prescribed_velocity_pressure(collect(pt), scenario)
        (vx, vy, 0.0)
    end
    pts = sd[triangle_center(sd), :dual_point]
    Vflat = ♭(sd, DualVectorField(vf.(pts)))

    topo = mesh_bundle.topology
    p0 = zeros(Float64, topo.num_vertices)
    T0 = map(topo.vertex_xyz) do xyz
        x, y = xyz[1], xyz[2]
        exp(-36.0 * ((x - 0.55)^2 + (y - 0.52)^2))
    end

    ComponentArray(; V = collect(Vflat), p = p0, T = T0)
end

function make_generate(params)
    function generate(sd, my_symbol; hodge = GeometricHodge())
        op =
            if my_symbol === :k
                x -> params.α * x
            else
                default_dec_generate(sd, my_symbol, hodge)
            end
        return (args...) -> op(args...)
    end
end

function assemble_simulator(; dim::Int = 2)
    D = expanded_coupled_multiphysics()
    sim_code = gensim(D; dimension = dim)
    Base.invokelatest(eval, sim_code)
end

function integrate_multiphysics(mesh_bundle, scenario::Symbol, params; t_end::Float64, n_frames::Int)
    sd = mesh_bundle.dual
    primal = mesh_bundle.primal

    sim = assemble_simulator(; dim = 2)
    gen = make_generate(params)
    fₘ = Base.invokelatest(sim, sd, gen)

    u₀ = initial_conditions(mesh_bundle, scenario)
    dec_params = (; Nu = params.ν, Invrho = params.invrho, Kappa = params.κ)

    tspan = (0.0, t_end)
    saveat = collect(range(tspan[1], tspan[2]; length = n_frames))
    prob = ODEProblem(fₘ, u₀, tspan, dec_params)
    sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1e-7, reltol = 1e-5)
    return sol, saveat
end

function stack_frames(sol, times, mesh_bundle)
    primal = mesh_bundle.primal
    topo = mesh_bundle.topology
    nvtx = topo.num_vertices
    nt = length(times)
    frames = Vector{Vector{Vector{Float64}}}(undef, nt)
    for it in 1:nt
        st = sol(times[it])
        vx, vy = vertex_velocity_proxy(primal, st.V)
        pr = Vector{Float64}(st.p)
        Tvec = Vector{Float64}(st.T)
        @assert length(pr) == nvtx && length(Tvec) == nvtx
        frames[it] = [[vx[j], vy[j], pr[j], Tvec[j]] for j in 1:nvtx]
    end
    frames
end

function persist_outputs(
    times,
    frames,
    mesh_bundle,
    scenario::Symbol,
    params,
    sol,
    out_json::Union{Nothing,String},
    out_jld::Union{Nothing,String},
)
    topo = mesh_bundle.topology
    np = length(times)
    mat_T = zeros(Float64, np, topo.num_vertices)
    mat_p = zeros(Float64, np, topo.num_vertices)
    mat_vx = zeros(Float64, np, topo.num_vertices)
    mat_vy = zeros(Float64, np, topo.num_vertices)
    for it in 1:np
        st = sol(times[it])
        vx, vy = vertex_velocity_proxy(mesh_bundle.primal, st.V)
        mat_vx[it, :] .= vx
        mat_vy[it, :] .= vy
        mat_p[it, :] .= st.p
        mat_T[it, :] .= st.T
    end

    base_payload = Dict{String,Any}(
        "schema" => "categorical_physics_engine_step1_v1",
        "single_source_of_truth" => "JSON contract mirrors Julia arrays; Python consumes after optional index translation.",
        "scenario" => string(scenario),
        "mesh_lx" => mesh_bundle.lx,
        "mesh_ly" => mesh_bundle.ly,
        "params" => Dict(
            "nu" => params.ν,
            "rho" => 1 / params.invrho,
            "invrho" => params.invrho,
            "alpha" => params.α,
            "kappa" => params.κ,
        ),
        "indexing_note" => "All incidence lists are **1-based Julia order**. Emit fixed vertex ordering matching vertex_xyz rows for deterministic PyG conversion (subtract 1 once).",
        "time" => collect(times),
        "num_nodes" => topo.num_vertices,
        "node_feature_names" => ["vx", "vy", "p", "T"],
        "node_features_time_series" => frames,
        "topology" => topology_payload(topo),
    )

    if out_json !== nothing
        mkpath(dirname(out_json))
        open(out_json, "w") do io
            JSON3.pretty(io, base_payload)
        end
        @info "Wrote JSON ground-truth contract" out_json
    end

    if out_jld !== nothing
        mkpath(dirname(out_jld))
        jldopen(out_jld, "w") do f
            f["time"] = collect(times)
            f["temperature"] = mat_T
            f["pressure"] = mat_p
            f["velocity_vertex_vx"] = mat_vx
            f["velocity_vertex_vy"] = mat_vy
            f["topology"] = topo
            f["scenario"] = string(scenario)
            f["params"] = (ν = params.ν, invrho = params.invrho, ρ = 1 / params.invrho, α = params.α, κ = params.κ)
        end
        @info "Wrote JLD2 archive" out_jld
    end
    return base_payload
end
