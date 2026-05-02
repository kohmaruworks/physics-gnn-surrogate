#!/usr/bin/env julia
"""
Step 1 runner — build DEC meshes, compose Decapodes diagrams, integrate with OrdinaryDiffEq,
and emit bilingual JSON/JLD2 payloads under `data/raw/`.

Usage (from `step1_initial_physics_def/`):

```bash
julia --project=. src/main.jl --scenario cylinder_wake --t-end 1.2
```
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "simulation.jl"))

function parse_cmdline()
    opts = Dict{Symbol,Any}(
        :scenario => :cylinder_wake,
        :t_end => 1.2,
        :nx => 36,
        :ny => 18,
        :n_frames => 73,
        :json => true,
        :jld2 => true,
        :nu => 0.02,
        :rho => 1.0,
        :alpha => 0.045,
        :kappa => 0.012,
    )
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--scenario" && i < length(ARGS)
            opts[:scenario] = Symbol(ARGS[i+1])
            i += 2
        elseif a == "--t-end" && i < length(ARGS)
            opts[:t_end] = parse(Float64, ARGS[i+1])
            i += 2
        elseif a == "--nx" && i < length(ARGS)
            opts[:nx] = parse(Int, ARGS[i+1])
            i += 2
        elseif a == "--ny" && i < length(ARGS)
            opts[:ny] = parse(Int, ARGS[i+1])
            i += 2
        elseif a == "--frames" && i < length(ARGS)
            opts[:n_frames] = parse(Int, ARGS[i+1])
            i += 2
        elseif a == "--nu" && i < length(ARGS)
            opts[:nu] = parse(Float64, ARGS[i+1])
            i += 2
        elseif a == "--rho" && i < length(ARGS)
            opts[:rho] = parse(Float64, ARGS[i+1])
            i += 2
        elseif a == "--alpha" && i < length(ARGS)
            opts[:alpha] = parse(Float64, ARGS[i+1])
            i += 2
        elseif a == "--kappa" && i < length(ARGS)
            opts[:kappa] = parse(Float64, ARGS[i+1])
            i += 2
        elseif a == "--no-json"
            opts[:json] = false
            i += 1
        elseif a == "--no-jld2"
            opts[:jld2] = false
            i += 1
        elseif a in ("-h", "--help")
            println("""
Usage: julia --project=. src/main.jl [options]

Options:
  --scenario NAME   cylinder_wake | heat_sink    (default: cylinder_wake)
  --t-end FLOAT     final time                   (default: 1.2)
  --nx INT          longitudinal grid cells      (default: 36)
  --ny INT          transverse grid cells        (default: 18)
  --frames INT      uniform save count           (default: 73)
  --nu FLOAT        kinematic viscosity scale    (default: 0.02)
  --rho FLOAT       density constant             (default: 1.0)
  --alpha FLOAT     thermal diffusivity constant (default: 0.045)
  --kappa FLOAT     pressure diffusion regularizer (default: 0.012)
  --no-json         skip JSON export
  --no-jld2         skip JLD2 export
""")
            exit(0)
        else
            @warn "Ignoring unknown CLI token" a
            i += 1
        end
    end
    opts
end

function main()
    opts = parse_cmdline()
    scenario = opts[:scenario]::Symbol

    bundle = multiphysics_bundle()
    @info "SummationDecapode fragments ready" typeof(bundle.navier_stokes) typeof(bundle.heat)

    mesh = build_scenario_mesh(scenario; nx = opts[:nx], ny = opts[:ny])
    params = (ν = opts[:nu], invrho = 1 / opts[:rho], α = opts[:alpha], κ = opts[:kappa])

    sol, times =
        integrate_multiphysics(mesh, scenario, params; t_end = opts[:t_end], n_frames = opts[:n_frames])
    frames = stack_frames(sol, times, mesh)

    raw_dir = joinpath(@__DIR__, "..", "data", "raw")
    mkpath(raw_dir)
    stamp = replace(string(scenario), ":" => "")
    json_path = opts[:json] ? joinpath(raw_dir, "ground_truth_$(stamp).json") : nothing
    jld_path = opts[:jld2] ? joinpath(raw_dir, "ground_truth_$(stamp).jld2") : nothing

    persist_outputs(times, frames, mesh, scenario, params, sol, json_path, jld_path)

    @info "Completed Step 1 integration" scenario nodes = mesh.topology.num_vertices steps = length(times)
    return nothing
end

main()
