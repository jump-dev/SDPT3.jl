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
    status::Union{Nothing, Int}
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
            nothing, NaN,
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
    optimizer.status = nothing
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
    empty!(optimizer.free_Cvar)
    empty!(optimizer.free_Cval)
    empty!(optimizer.nneg_Cvar)
    empty!(optimizer.nneg_Cval)
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

function MOI.optimize!(optimizer::Optimizer)
    options = optimizer.options
    if optimizer.silent
        options = copy(options)
        options[:printlevel] = 0
    end

    blk = Matrix{Any}(undef, 0, 2)
    blkt_vec = Any[]
    m = length(optimizer.b)
    At = SparseArrays.SparseMatrixCSC{Float64,Int}[]
    # FIXME I get a strange failure with sparse vectors, need to investigate
    C = Vector{Float64}[]
    if !iszero(optimizer.num_nneg)
        push!(At, sparse(optimizer.nneg_Avar, optimizer.nneg_Acon, optimizer.nneg_Aval, optimizer.num_nneg, m))
        push!(C, Vector(sparsevec(optimizer.nneg_Cvar, optimizer.nneg_Cval, optimizer.num_nneg)))
        blk = vcat(blk, ["l" optimizer.num_nneg])
    end
    if !iszero(optimizer.num_free)
        push!(At, sparse(optimizer.free_Avar, optimizer.free_Acon, optimizer.free_Aval, optimizer.num_free, m))
        push!(C, Vector(sparsevec(optimizer.free_Cvar, optimizer.free_Cval, optimizer.num_free)))
        blk = vcat(blk, ["u" optimizer.num_free])
    end

    options = optimizer.options
    if optimizer.silent
        options = copy(options)
        options[:printlevel] = 0
    end

    obj, X, y, Z, optimizer.info, optimizer.runhist = sdpt3(
        blk, At, C, optimizer.b; options...)

    optimizer.primal_objective_value, optimizer.dual_objective_value = obj
    k = 0
    if iszero(optimizer.num_nneg)
        empty!(optimizer.nneg_X)
        empty!(optimizer.nneg_Z)
    else
        k += 1
        optimizer.nneg_X = X[k]
        optimizer.nneg_Z = Z[k]
    end
    if iszero(optimizer.num_free)
        empty!(optimizer.free_X)
        empty!(optimizer.free_Z)
    else
        k += 1
        optimizer.free_X = X[k]
        optimizer.free_Z = Z[k]
    end
    optimizer.y = y
    optimizer.status = optimizer.info["termcode"]
    optimizer.solve_time = optimizer.info["cputime"]
end

const RAW_STATUS = [
    "norm(X) or norm(Z) diverging",
    "dual   problem is suspected to be infeasible",
    "primal problem is suspected to be infeasible",
    "max(relative gap,infeasibility) < gaptol",
    "relative gap < infeasibility",
    "lack of progress in predictor or corrector",
    "X or Z not positive definite",
    "difficulty in computing predictor or corrector direction",
    "progress in relative gap or infeasibility is bad",
    "maximum number of iterations reached",
    "primal infeasibility has deteriorated too much",
    "progress in relative gap has deteriorated",
    "lack of progress in infeasibility"
]
function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    if optimizer.status === nothing
        throw(MOI.OptimizeNotCalled())
    else
        return RAW_STATUS[4 - optimizer.status]
    end
end

const TERMINATION_STATUS = [
    MOI.NUMERICAL_ERROR,
    MOI.DUAL_INFEASIBLE,
    MOI.INFEASIBLE,
    MOI.OPTIMAL,
    MOI.OTHER_ERROR, # TODO what does `relative gap < infeasibility` mean ?
    MOI.SLOW_PROGRESS,
    MOI.NUMERICAL_ERROR,
    MOI.NUMERICAL_ERROR,
    MOI.SLOW_PROGRESS,
    MOI.ITERATION_LIMIT,
    MOI.NUMERICAL_ERROR,
    MOI.NUMERICAL_ERROR,
    MOI.SLOW_PROGRESS
]
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if optimizer.status === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    else
        return TERMINATION_STATUS[4 - optimizer.status]
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.PrimalStatus)
    status = optimizer.status
    if status == 0
        return MOI.FEASIBLE_POINT
    elseif status == 2
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        # TODO is there solution available in some case here ?
        return MOI.NO_SOLUTION
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.DualStatus)
    status = optimizer.status
    if status == 0
        return MOI.FEASIBLE_POINT
    elseif status == 1
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        # TODO is there solution available in some case here ?
        return MOI.NO_SOLUTION
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
