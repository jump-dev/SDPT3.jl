using LinearAlgebra # For rmul!

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
const AFFEQ = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Cdouble}, MOI.EqualTo{Cdouble}}

@enum VariableType FREE NNEG SOC PSD

struct VariableInfo
    variable_type::VariableType
    cone_index::Int
    index_in_cone::Int
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    b::Vector{Float64}

    variable_info::Vector{VariableInfo}

    num_nneg::Int
    nneg_Cvar::Vector{Int}
    nneg_Cval::Vector{Float64}
    nneg_Avar::Vector{Int}
    nneg_Acon::Vector{Int}
    nneg_Aval::Vector{Float64}

    num_free::Int
    free_Cvar::Vector{Int}
    free_Cval::Vector{Float64}
    free_Avar::Vector{Int}
    free_Acon::Vector{Int}
    free_Aval::Vector{Float64}

    objective_sign::Int
    objective_constant::Float64

    primal_objective_value::Float64
    dual_objective_value::Float64
    free_X::Vector{Float64}
    nneg_X::Vector{Float64}
    y::Vector{Float64}
    free_Z::Vector{Float64}
    nneg_Z::Vector{Float64}
    info::Dict{String, Any}
    runhist::Dict{String, Any}
    status::Int
    solve_time::Float64

    silent::Bool
    options::Dict{Symbol, Any}
    function Optimizer(; kwargs...)
        optimizer = new(
            Float64[], VariableInfo[],
            0, Int[], Float64[], Int[], Int[], Float64[],
            0, Int[], Float64[], Int[], Int[], Float64[],
            1, 0.0,
            NaN, NaN, Float64[], Float64[], Float64[], Float64[], Float64[],
            Dict{String, Any}(), Dict{String, Any}(),
            -1, NaN,
            false, Dict{Symbol, Any}())
        for (key, value) in kwargs
            MOI.set(optimizer, MOI.RawParameter(key), value)
        end
        return optimizer
    end
end

function MOI.supports(optimizer::Optimizer, param::MOI.RawParameter)
    return param.name in ALLOWED_OPTIONS
end
function MOI.set(optimizer::Optimizer, param::MOI.RawParameter, value)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    optimizer.options[param.name] = value
end
function MOI.get(optimizer::Optimizer, param::MOI.RawParameter)
    # TODO: This gives a poor error message if the name of the parameter is invalid.
    return optimizer.options[param.name]
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
end
MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

MOI.get(::Optimizer, ::MOI.SolverName) = "SDPT3"
# TODO this is a copy-paste from CSDP at the moment
const RAW_STATUS = [
    "Problem solved to optimality.",
    "Problem is primal infeasible.",
    "Problem is dual infeasible."
]

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return RAW_STATUS[optimizer.status + 1]
end
function MOI.get(optimizer::Optimizer, ::MOI.SolveTime)
    return optimizer.solve_time
end

function MOI.is_empty(optimizer::Optimizer)
    return isempty(optimizer.b) &&
        isempty(optimizer.variable_info) &&
        iszero(optimizer.num_nneg) &&
        iszero(optimizer.num_free) &&
        isone(optimizer.objective_sign) &&
        iszero(optimizer.objective_constant)
end

function MOI.empty!(optimizer::Optimizer)
    empty!(optimizer.b)

    empty!(optimizer.variable_info)

    optimizer.num_nneg = 0
    empty!(optimizer.nneg_Cvar)
    empty!(optimizer.nneg_Cval)
    empty!(optimizer.nneg_Avar)
    empty!(optimizer.nneg_Acon)
    empty!(optimizer.nneg_Aval)

    optimizer.num_free = 0
    empty!(optimizer.free_Cvar)
    empty!(optimizer.free_Cval)
    empty!(optimizer.free_Avar)
    empty!(optimizer.free_Acon)
    empty!(optimizer.free_Aval)

    optimizer.objective_sign = 1
    optimizer.objective_constant = 0.0

    optimizer.primal_objective_value = NaN
    optimizer.dual_objective_value = NaN
    empty!(optimizer.free_X)
    empty!(optimizer.nneg_X)
    empty!(optimizer.y)
    empty!(optimizer.free_Z)
    empty!(optimizer.nneg_Z)
    empty!(optimizer.info)
    empty!(optimizer.runhist)
    optimizer.status = -1
    optimizer.solve_time = NaN
end

function MOI.supports(
    optimizer::Optimizer,
    ::Union{MOI.ObjectiveSense,
            MOI.ObjectiveFunction{<:Union{MOI.SingleVariable,
                                          MOI.ScalarAffineFunction{Cdouble}}}})
    return true
end

const SupportedSets = Union{MOI.Nonnegatives, MOI.SecondOrderCone, MOI.PositiveSemidefiniteConeTriangle}
function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.VectorOfVariables},
    ::Type{<:SupportedSets})
    return true
end
function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.ScalarAffineFunction{Cdouble}},
    ::Type{MOI.EqualTo{Cdouble}})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end
MOIU.supports_default_copy_to(::Optimizer, copy_names::Bool) = !copy_names

# Variables
function MOI.add_variable(optimizer::Optimizer)
    optimizer.num_free += 1
    push!(optimizer.variable_info, VariableInfo(FREE, 1, optimizer.num_free))
    return MOI.VariableIndex(length(optimizer.variable_info))
end
function MOI.add_variables(optimizer::Optimizer, n::Integer)
    return [MOI.add_variable(optimizer) for i in 1:n]
end
function add_nonneg_variable(optimizer::Optimizer)
    optimizer.num_nneg += 1
    push!(optimizer.variable_info, VariableInfo(NNEG, 1, optimizer.num_nneg))
    return MOI.VariableIndex(length(optimizer.variable_info))
end
function MOI.add_constrained_variables(optimizer::Optimizer, set::MOI.Nonnegatives)
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(optimizer.num_nneg + 1)
    return [add_nonneg_variable(optimizer) for i in 1:MOI.dimension(set)], ci
end

# Objective
function MOI.set(optimizer::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    # To be sure that it is done before load(optimizer, ::ObjectiveFunction, ...), we do it in allocate
    sign = sense == MOI.MAX_SENSE ? -1 : 1
    if sign != optimizer.objective_sign
        rmul!(optimizer.free_Cval, -1)
        rmul!(optimizer.nneg_Cval, -1)
    end
    optimizer.objective_sign = sign
end
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
function MOI.set(optimizer::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
                 func::MOI.ScalarAffineFunction{Float64})
    optimizer.objective_constant = MOI.constant(func)
    for term in func.terms
        info = optimizer.variable_info[term.variable_index.value]
        if info.variable_type == FREE
            push!(optimizer.free_Cvar, info.index_in_cone)
            push!(optimizer.free_Cval, optimizer.objective_sign * term.coefficient)
        elseif info.variable_type == NNEG
            push!(optimizer.nneg_Cvar, info.index_in_cone)
            push!(optimizer.nneg_Cval, optimizer.objective_sign * term.coefficient)
        else
            @assert false
        end
    end
end
function MOI.set(optimizer::Optimizer, ::MOI.ObjectiveFunction, f::MOI.SingleVariable)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            convert(MOI.ScalarAffineFunction{Float64}, f))
end

# Constraints
function MOI.add_constraint(optimizer::Optimizer, func::MOI.ScalarAffineFunction{Float64}, set::MOI.EqualTo{Float64})
    if !iszero(MOI.constant(func))
        # We use the fact that the initial function constant was zero to
        # implement getters for `MOI.ConstraintPrimal`.
        throw(MOI.ScalarFunctionConstantNotZero{
             Float64, typeof(func), typeof(set)}(MOI.constant(func)))
    end
    push!(optimizer.b, MOI.constant(set))
    con = length(optimizer.b)
    for term in func.terms
        info = optimizer.variable_info[term.variable_index.value]
        if info.variable_type == FREE
            push!(optimizer.free_Avar, info.index_in_cone)
            push!(optimizer.free_Acon, con)
            push!(optimizer.free_Aval, term.coefficient)
        elseif info.variable_type == NNEG
            push!(optimizer.nneg_Avar, info.index_in_cone)
            push!(optimizer.nneg_Acon, con)
            push!(optimizer.nneg_Aval, term.coefficient)
        else
            @assert false
        end
    end
    return AFFEQ(con)
end

_vec(x::Vector) = x
_vec(x::Float64) = [x]

function MOI.optimize!(optimizer::Optimizer)
    options = optimizer.options
    if optimizer.silent
        options = copy(options)
        options[:printlevel] = 0
    end

    blk = [
        "l" optimizer.num_nneg
        "u" optimizer.num_free
    ]
    m = length(optimizer.b)
    nneg_A = sparse(optimizer.nneg_Avar, optimizer.nneg_Acon, optimizer.nneg_Aval, optimizer.num_nneg, m)
    free_A = sparse(optimizer.free_Avar, optimizer.free_Acon, optimizer.free_Aval, optimizer.num_free, m)
    At = [nneg_A, free_A]
    nneg_C = Vector(sparsevec(optimizer.nneg_Cvar, optimizer.nneg_Cval, optimizer.num_nneg))
    free_C = Vector(sparsevec(optimizer.free_Cvar, optimizer.free_Cval, optimizer.num_free))
    C = [nneg_C, free_C]

    # TODO use options
    obj, X, y, Z, optimizer.info, optimizer.runhist = sdpt3(
        blk, At, C, optimizer.b)
    optimizer.primal_objective_value, optimizer.dual_objective_value = obj
    optimizer.nneg_X = _vec(X[1])
    optimizer.free_X = _vec(X[2])
    optimizer.y = _vec(y)
    optimizer.nneg_Z = _vec(Z[1])
    optimizer.free_Z = _vec(Z[2])
    optimizer.status = optimizer.info["termcode"]
    optimizer.solve_time = optimizer.info["cputime"]
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    status = optimizer.status
    if status == -1
        return MOI.OPTIMIZE_NOT_CALLED
    elseif status == 0
        return MOI.OPTIMAL
    elseif status == 1
        return MOI.INFEASIBLE
    elseif status == 2
        return MOI.DUAL_INFEASIBLE
    else
        error("Unrecognized status $status.")
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.PrimalStatus)
    status = optimizer.status
    if status == -1
        return MOI.NO_SOLUTION
    elseif status == 0
        return MOI.FEASIBLE_POINT
    elseif status == 1
        return MOI.NO_SOLUTION
    elseif status == 2
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        error("Unrecognized status $status.")
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.DualStatus)
    status = optimizer.status
    if status == -1
        return MOI.NO_SOLUTION
    elseif status == 0
        return MOI.FEASIBLE_POINT
    elseif status == 1
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == 2
        return MOI.NO_SOLUTION
    else
        error("Unrecognized status $status.")
    end
end

MOI.get(m::Optimizer, ::MOI.ResultCount) = 1
function MOI.get(m::Optimizer, ::MOI.ObjectiveValue)
    return m.objective_sign * m.primal_objective_value + m.objective_constant
end
function MOI.get(m::Optimizer, ::MOI.DualObjectiveValue)
    # FIXME this is the primal obj
    return m.objective_sign * m.dual_objective_value + m.objective_constant
end

function MOI.get(optimizer::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    info = optimizer.variable_info[vi.value]
    if info.variable_type == FREE
        return optimizer.free_X[info.index_in_cone]
    elseif info.variable_type == NNEG
        return optimizer.nneg_X[info.index_in_cone]
    else
        @assert false
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where S<:SupportedSets
    info = optimizer.variable_info[ci.value]
    if info.variable_type == FREE
        @assert false
    elseif info.variable_type == NNEG
        # TODO need to figure out dim
        return [optimizer.nneg_X[info.index_in_cone]]
    else
        @assert false
    end
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal, ci::AFFEQ)
    return optimizer.b[ci.value]
end

function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where S<:SupportedSets
    info = optimizer.variable_info[ci.value]
    if info.variable_type == FREE
        @assert false
    elseif info.variable_type == NNEG
        # TODO need to figure out dim
        return [optimizer.nneg_Z[info.index_in_cone]]
    else
        @assert false
    end
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual, ci::AFFEQ)
    return optimizer.y[ci.value]
end
