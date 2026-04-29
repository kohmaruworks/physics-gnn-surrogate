# Decapodes: N–S（渦度–流線関数雛形）と熱（0-form）を別定義し、operadic 合成の骨格を返す。
# 依存: Decapodes, Catlab
#
# 注: 完全な可圧/境界込み N–S+熱は `generate`・境界演算子と併せて拡張する。ここは合成パターンと式の枠を固定する。

using DiagrammaticEquations
using Decapodes
using Decapodes: @decapode, oapply, Open
using Catlab
using Catlab.Programs: @relation
using Catlab.CategoricalAlgebra: apex

"""非圧縮性 Euler 系（渦度形式, Poisson 補正付き）— Decapodes 公式例に沿う雛形。"""
function decapode_fluid_navier_stokes_inviscid_vorticity()
    @decapode begin
        d𝐮::DualForm2
        𝐮::DualForm1
        ψ::Form0

        ψ == Δ⁻¹(⋆(d𝐮))
        𝐮 == ⋆(d(ψ))

        ∂ₜ(d𝐮) == (-1) * ∘(♭♯, ⋆₁, d̃₁)(∧ᵈᵖ₁₀(𝐮, ⋆(d𝐮)))
    end
end

"""熱: 拡散 + 移流用に速度 1-form `𝐮` スロット（移流項は下流で Lie 導来等に接続）。"""
function decapode_heat_advection_diffusion()
    @decapode begin
        T::Form0
        𝐮::Form1
        (α)::Constant

        ∂ₜ(T) == α * Δ(T)
    end
end

# ---- 速度 𝐮 を共有する「流体 + 熱」operadic 合成（簡略化した 1-form モメンタム枠 + 拡散）----
function _decapode_fluid_momentum_simplified()
    @decapode begin
        𝐮::Form1
        p::Form0
        (ν, ρ)::Constant

        ∂ₜ(𝐮) == ν * Δ(𝐮) - d(p) / ρ
    end
end

function _decapode_heat_with_shared_velocity()
    @decapode begin
        T::Form0
        𝐮::Form1
        (α)::Constant

        ∂ₜ(T) == α * Δ(T)
    end
end

"""`𝐮`（Form1）を **共有junction** として operad 合成した連立 Decapode（熱は現段階拡散のみ; 移流項は拡張点）。"""
function decapode_compose_fluid_heat_shared_velocity()
    F = _decapode_fluid_momentum_simplified()
    H = _decapode_heat_with_shared_velocity()
    # Catlab 0.17+: 外枠のポートは @relation () begin（空）で、共有変数は箱間の同一ラベルで接続
    pat = @relation () begin
        fluid(𝐮, p)
        heat(𝐮, T)
    end
    cospan = oapply(pat, [Open(F, [:𝐮, :p]), Open(H, [:𝐮, :T])])
    apex(cospan)
end

# ---- Operadic 合成例（Klausmeier 型 UWD; 公式チュートリアル互換）----
function _demo_hydrodynamics_block()
    @decapode begin
        (n, w)::DualForm0
        dX::Form1
        (a, ν)::Constant

        ∂ₜ(w) == a - w - w * n^2 + ν * L(dX, w)
    end
end

function _demo_reaction_block()
    @decapode begin
        (n, w)::DualForm0
        m::Constant

        ∂ₜ(n) == w * n^2 - m * n + Δ(n)
    end
end

"""2 ブロックの undirected wiring + `Open` 合成（apex = 連立 Decapode）。"""
function decapode_compose_operadic_demo()
    H = _demo_hydrodynamics_block()
    P = _demo_reaction_block()
    pat = @relation () begin
        hydro(n, w)
        reaction(n, w)
    end
    cospan = oapply(pat, [Open(H, [:n, :w]), Open(P, [:n, :w])])
    apex(cospan)
end

"""
速度 `𝐮` を共有する N–S 雛形 + 熱雛形の **結合方針**:
- 厳密な一つの apex に束ねるには、同じ `𝐮` 名の変数を共有する大きな UWD を追加で定義する（メッシュ幅・演算子解決の段階と合わせる）。
- 本ステップでは `bundle` として (流体 decapode, 熱 decapode, 合成例 apex) を返し、下流の `03_simulation` では toy 積分でパイプライン検証に回す。
"""
function build_coupled_multiphysics_model()
    return (
        navier_stokes = decapode_fluid_navier_stokes_inviscid_vorticity(),
        heat = decapode_heat_advection_diffusion(),
        coupled_fluid_heat = decapode_compose_fluid_heat_shared_velocity(),
        operadic_compose_demo = decapode_compose_operadic_demo(),
    )
end
