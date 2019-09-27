using Test

using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

import SDPT3
const OPTIMIZER = SDPT3.Optimizer()
MOI.set(OPTIMIZER, MOI.Silent(), true)

@testset "SolverName" begin
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "SDPT3"
end

@testset "supports_default_copy_to" begin
    @test MOIU.supports_default_copy_to(OPTIMIZER, false)
    @test !MOIU.supports_default_copy_to(OPTIMIZER, true)
end

# UniversalFallback is needed for starting values, even if they are ignored by SDPT3
const CACHE = MOIU.UniversalFallback(MOIU.Model{Float64}())
const CACHED = MOIU.CachingOptimizer(CACHE, OPTIMIZER)
const BRIDGED = MOIB.full_bridge_optimizer(CACHED, Float64)
const CONFIG = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

@testset "Options" begin
    param = MOI.RawParameter(:bad_option)
    err = MOI.UnsupportedAttribute(param)
    @test_throws err SDPT3.Optimizer(bad_option = 1)
end

@testset "Unit" begin
    MOIT.unittest(BRIDGED, CONFIG, [
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
    # See explanation in `MOI/test/Bridges/lazy_bridge_OPTIMIZER.jl`.
    # This is to avoid `Variable.VectorizeBridge` which does not support
    # `ConstraintSet` modification.
    MOIB.remove_bridge(BRIDGED, MOIB.Constraint.ScalarSlackBridge{Float64})
    MOIT.contlineartest(BRIDGED, CONFIG, String[
        # Throws error: total dimension of C should be > length(b)
        "linear15",
        "partial_start"
    ])
end
@testset "Continuous Quadratic" begin
    MOIT.contquadratictest(BRIDGED, CONFIG, [
        # FIXME CachingOptimizer does not implement deletion of vector of variables and uses the fallback. Need upstream MOI fix.
        "qp2", "qp3",
        # Non-convex
        "ncqcp",
        # Quadratic function not strictly convex
        "socp"])
end
@testset "Continuous Conic" begin
    MOIT.contconictest(BRIDGED, CONFIG, [
        # `MOI.OTHER_ERROR`
        "lin2f",
        # `MOI.NUMERICAL_ERROR`
        "lin4",
        # `MOI.NUMERICAL_ERROR`: should be fixed by a RSOC->SOC Variable bridge
        "rotatedsoc2",
        # `ExponentialCone` and `PowerCone` not supported.
        "exp", "dualexp", "pow", "dualpow", "logdet",
        # `RootDetConeSquare` -> `RootDetConeTriangle` bridge missing.
        "rootdets"
    ])
end
