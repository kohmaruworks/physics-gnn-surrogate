#!/usr/bin/env julia
"""
Step 2 runner — load Step 1 cylinder wake ground truth (JLD2 or JSON), export heterogeneous DEC graph (JSON Contract V2).

Usage (from `step2_heterogeneous_contract/`):

```bash
julia --project=. src/main.jl
```
"""

using Pkg

const ROOT = dirname(@__DIR__)
Pkg.activate(ROOT)

include(joinpath(ROOT, "src", "export_hetero_json.jl"))

function main()
    raw_dir = joinpath(ROOT, "..", "step1_initial_physics_def", "data", "raw")
    step1_jld = joinpath(raw_dir, "ground_truth_cylinder_wake.jld2")
    step1_json = joinpath(raw_dir, "ground_truth_cylinder_wake.json")
    step1_path = isfile(step1_jld) ? step1_jld : step1_json
    isfile(step1_path) || error("Step 1 data missing; expected $(step1_jld) or $(step1_json)")

    out_json = joinpath(ROOT, "data", "v2_contract", "hetero_cylinder_wake_t0.35.json")
    export_hetero_json_v2(step1_path, out_json; t_target = 0.35)
    return nothing
end

main()
