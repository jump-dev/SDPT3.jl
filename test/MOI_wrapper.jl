module TestSDPT3

using Test
using MathOptInterface
import SDPT3

const MOI = MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_solver_name()
    @test MOI.get(SDPT3.Optimizer(), MOI.SolverName()) == "SDPT3"
end

function test_supports_incremental_interface()
    @test MOI.supports_incremental_interface(SDPT3.Optimizer())
end

function test_options()
    optimizer = SDPT3.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("printlevel"), 1)
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("printlevel")) == 1

    param = MOI.RawOptimizerAttribute("bad_option")
    err = MOI.UnsupportedAttribute(param)
    @test_throws err MOI.set(optimizer, MOI.RawOptimizerAttribute("bad_option"), 1)
end

function test_runtests()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                SDPT3.Optimizer(),
            ),
            Float64,
        ),
        # This does not work as with some modifications, the bridges with try
        # getting `ConstraintFunction` which is not supported by SDPT3
        #MOI.instantiate(SDPT3.Optimizer, with_bridge_type=Float64),
    )
    # `Variable.ZerosBridge` makes dual needed by some tests fail.
    MOI.Bridges.remove_bridge(model.optimizer, MathOptInterface.Bridges.Variable.ZerosBridge{Float64})
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            rtol = 1e-4,
            atol = 1e-4,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
                MOI.SolverVersion,
            ],
        ),
        exclude = String[
            # Expected test failures:
            #"test_attribute_SolverVersion",
            # TODO Remove when https://github.com/jump-dev/MathOptInterface.jl/issues/1758 is fixed
            "test_model_copy_to_UnsupportedAttribute",
            # `NUMERICAL_ERROR`
            "test_conic_linear_INFEASIBLE_2",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan",
            "test_modification_const_vectoraffine_zeros",
            "test_constraint_ScalarAffineFunction_EqualTo",
            # `OTHER_ERROR`
            "test_conic_linear_VectorAffineFunction_2",
            "test_conic_GeometricMeanCone_VectorOfVariables_3",
            #   ArgumentError: SDPT3 does not support problems with no constraint.
            "test_solve_optimize_twice",
            "test_solve_result_index",
            "test_quadratic_nonhomogeneous",
            "test_quadratic_integration",
            "test_objective_ObjectiveFunction_constant",
            "test_objective_ObjectiveFunction_VariableIndex",
            "test_objective_FEASIBILITY_SENSE_clears_objective",
            "test_modification_transform_singlevariable_lessthan",
            "test_modification_set_singlevariable_lessthan",
            "test_modification_delete_variables_in_a_batch",
            "test_modification_delete_variable_with_single_variable_obj",
            "test_modification_const_scalar_objective",
            "test_modification_coef_scalar_objective",
            "test_attribute_RawStatusString",
            "test_attribute_SolveTimeSec",
            "test_objective_ObjectiveFunction_blank",
            "test_objective_ObjectiveFunction_duplicate_terms",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # ArgumentError: SDPT3 does not support equality constraints with no term
            "test_linear_VectorAffineFunction_empty_row",
            "test_conic_PositiveSemidefiniteConeTriangle",
            # FIXME
            #  Expression: ≈(MOI.get(model, MOI.ConstraintPrimal(), c2), 0, atol = atol, rtol = rtol)
            #  Evaluated: 1.7999998823840366 ≈ 0 (atol=0.0001, rtol=0.0001)
            "test_linear_FEASIBILITY_SENSE",
            # FIXME
            #  Error using pretransfo (line 149)
            #  Size b mismatch
            "test_conic_SecondOrderCone_negative_post_bound_ii",
            "test_conic_SecondOrderCone_negative_post_bound_iii",
            "test_conic_SecondOrderCone_no_initial_bound",
            # TODO SDPT3 just returns an infinite ObjectiveValue
            "test_unbounded_MIN_SENSE",
            "test_unbounded_MIN_SENSE_offset",
            "test_unbounded_MAX_SENSE",
            "test_unbounded_MAX_SENSE_offset",
            # TODO SDPT3 just returns an infinite DualObjectiveValue
            "test_infeasible_MAX_SENSE",
            "test_infeasible_MAX_SENSE_offset",
            "test_infeasible_MIN_SENSE",
            "test_infeasible_MIN_SENSE_offset",
            "test_infeasible_affine_MAX_SENSE",
            "test_infeasible_affine_MAX_SENSE_offset",
            "test_infeasible_affine_MIN_SENSE",
            "test_infeasible_affine_MIN_SENSE_offset",
            # TODO investigate
            "test_conic_GeometricMeanCone_VectorAffineFunction_2",
            "test_conic_GeometricMeanCone_VectorOfVariables_2",
            "test_objective_qp_ObjectiveFunction_edge_cases",
            "test_objective_qp_ObjectiveFunction_zero_ofdiag",
            "test_variable_solve_with_lowerbound",
            # FIXME
            # test_linear_DUAL_INFEASIBLE_2: Test Failed at /home/blegat/.julia/packages/MathOptInterface/IIN1o/src/Test/test_linear.jl:1514
            #  Expression: MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE || MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE_OR_UNBOUNDED
            "test_linear_DUAL_INFEASIBLE_2",
        ],
    )
    return
end

end  # module

TestSDPT3.runtests()
