using Test

using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

import SDPT3
const optimizer = SDPT3.Optimizer()
MOI.set(optimizer, MOI.Silent(), true)

@testset "SolverName" begin
    @test MOI.get(optimizer, MOI.SolverName()) == "SDPT3"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_default_copy_to(optimizer, false)
    @test !MOIU.supports_default_copy_to(optimizer, true)
end

# UniversalFallback is needed for starting values, even if they are ignored by SDPT3
const cache = MOIU.UniversalFallback(MOIU.Model{Float64}())
const cached = MOIU.CachingOptimizer(cache, optimizer)
const bridged = MOIB.full_bridge_optimizer(cached, Float64)
const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

@testset "Options" begin
    param = MOI.RawParameter(:bad_option)
    err = MOI.UnsupportedAttribute(param)
    @test_throws err SDPT3.Optimizer(bad_option = 1)
end

@testset "Unit" begin
    MOIT.unittest(bridged, config, [
        # Get `termcode` -1, i.e. "relative gap < infeasibility".
        "solve_blank_obj",
        # Get `termcode` 3, i.e. "norm(X) or norm(Z) diverging".
        "solve_affine_equalto",
        # Fails because there is no constraint.
        "solve_unbounded_model",
        # `TimeLimitSec` not supported.
        "time_limit_sec",
        # Quadratic functions are not supported
        "solve_qcp_edge_cases", "solve_qp_edge_cases",
        # Integer and ZeroOne sets are not supported
        "solve_integer_edge_cases", "solve_objbound_edge_cases",
        "solve_zero_one_with_bounds_1",
        "solve_zero_one_with_bounds_2",
        "solve_zero_one_with_bounds_3"])
end
@testset "Continuous Linear" begin
    # See explanation in `MOI/test/Bridges/lazy_bridge_optimizer.jl`.
    # This is to avoid `Variable.VectorizeBridge` which does not support
    # `ConstraintSet` modification.
    MOIB.remove_bridge(bridged, MOIB.Constraint.ScalarSlackBridge{Float64})
    MOIT.contlineartest(bridged, config, String[
        # Throws error: total dimension of C should be > length(b)
        "linear15",
        "partial_start"
    ])
end
