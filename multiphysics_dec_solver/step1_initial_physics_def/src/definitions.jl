"""
Piecewise `SummationDecapode` definitions + operadic composition (`Open`, `oapply`) for
coupled fluid momentum / pressure / scalar temperature on shared primal velocity `V`.

Continuous targets (reference):

\\[
\\partial_t \\mathbf{u} + (\\mathbf{u}\\cdot\\nabla)\\mathbf{u}
 = -\\tfrac{1}{\\rho}\\nabla p + \\nu \\nabla^2 \\mathbf{u} + \\mathbf{f}, \\quad \\nabla\\cdot\\mathbf{u}=0,
\\qquad
\\partial_t T + \\nabla\\cdot(T \\mathbf{u}) = \\alpha \\nabla^2 T.
\\]

Discrete synthesis follows `examples/sw/sw.jl`: thermal conductivity in `∘(d₀, k, ⋆₁)` is wired **only** through `generate` (`:k`), never `k::Constant` (which would bind `__p__.k` as a scalar and break the composed operator). Fluid coefficients `(Nu, Invrho, Kappa)` are `Constant` symbols mapped from `ODEProblem` parameters `__p__`.
"""

using Catlab
using Catlab.CategoricalAlgebra: apex
using Catlab.Programs: @relation
using Decapodes
using DiagrammaticEquations
using DiagrammaticEquations.Deca

"""Navier–Stokes fragment (`SummationDecapode`): Stokes momentum + pressure diffusion."""
function summation_navier_stokes_momentum()
    @decapode begin
        V::Form1
        p::Form0
        ∇p::Form1
        (Nu, Invrho, Kappa)::Constant
        ∇p == d₀(p)
        ∂ₜ(V) == Nu * Δ(V) - Invrho * ∇p
        ∂ₜ(p) == Kappa * Δ(p)
    end
end

"""Thermal advection–diffusion (`SummationDecapode`) sharing primal velocity `V`.

`k` appears only inside `∘(d₀, k, ⋆₁)` and is supplied by `generate` (Diffusion coefficient scaling), not `__p__` — mirroring `examples/sw/sw.jl` AdvDiff.
"""
function summation_heat_advection_diffusion()
    @decapode begin
        T::Form0
        V::Form1
        ϕ::Form1
        ϕ₁::Form1
        ϕ₂::Form1
        ϕ₁ == ∘(d₀, k, ⋆₁)(T)
        ϕ₂ == -∧₀₁(T, V)
        ϕ == plus(ϕ₁, ϕ₂)
        ∂ₜ(T) == ∘(dual_d₁, ⋆₀⁻¹)(ϕ)
    end
end

"""Operadic wiring (`OpenSummationDecapode`): fuse `V` across fluid + thermal blocks."""
function compose_ns_heat_operadic()
    pat = @relation () begin
        fluid(V, p)
        heat(T, V)
    end
    cospan = oapply(pat, [
        Open(summation_navier_stokes_momentum(), [:V, :p]),
        Open(summation_heat_advection_diffusion(), [:T, :V]),
    ])
    apex(cospan)
end

"""Unified coupling (`SummationDecapode`) for `gensim`.

Scalar coefficients `(Nu, Invrho, Kappa)` use `Constant` → `__p__`; thermal `k` in `∘(d₀, k, ⋆₁)` uses `generate` only (see `sw.jl`).
"""
function summation_coupled_ns_heat_executable()
    q = quote
        V::Form1
        p::Form0
        T::Form0
        ∇p::Form1
        ϕ::Form1
        ϕ₁::Form1
        ϕ₂::Form1
        (Nu, Invrho, Kappa)::Constant
        ∇p == d₀(p)
        ∂ₜ(V) == Nu * Δ(V) - Invrho * ∇p
        ∂ₜ(p) == Kappa * Δ(p)
        ϕ₁ == ∘(d₀, k, ⋆₁)(T)
        ϕ₂ == -∧₀₁(T, V)
        ϕ == plus(ϕ₁, ϕ₂)
        ∂ₜ(T) == ∘(dual_d₁, ⋆₀⁻¹)(ϕ)
    end
    SummationDecapode(parse_decapode(q))
end

"""`expand_operators` — inference handled inside `gensim`."""
function expanded_coupled_multiphysics()
    expand_operators(summation_coupled_ns_heat_executable())
end

function multiphysics_bundle()
    (
        navier_stokes = summation_navier_stokes_momentum(),
        heat = summation_heat_advection_diffusion(),
        coupled_operadic = compose_ns_heat_operadic(),
        coupled_executable = summation_coupled_ns_heat_executable(),
        coupled_expanded = expanded_coupled_multiphysics(),
    )
end
