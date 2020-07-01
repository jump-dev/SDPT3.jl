module SDPT3

using SparseArrays
using MATLAB

export sdpt3

#function dim(c::Char, n::Float64)
#    if c == 's'
#        return div(n * (n + 1), 2)
#    else
#        return n
#    end
#end

# See Solver/sqlparameters.m
const ALLOWED_OPTIONS = [
    "vers",
    "gam",
    "predcorr",
    "expon",
    "gaptol",
    "inftol",
    "steptol",
    "maxit",
    "printlevel",
    "stoplevel",
    "scale_data",
    "rmdepconstr",
    "smallblkdim",
    "parbarrier",
    "schurfun",
    "schurfun_par"
]

# `SparseMatrixCSC` is returned in SumOfSquares.jl test `sos_horn`
_array(x::AbstractMatrix) = x
_array(x::Vector) = x
_array(x::Union{Float64, Complex{Float64}}) = [x]

# TODO log in objective, OPTION, initial iterates X0, y0, Z0
# Solve the primal/dual pair
# min c'x,      max b'y
# s.t. Ax = b,   c - A'x ∈ K
#       x ∈ K
function sdpt3(blk::Matrix,
               At::Vector{<:Union{Matrix{<:Union{Float64, Complex{Float64}}},
                                  SparseMatrixCSC{<:Union{Float64, Complex{Float64}}}}},
               C::Vector{<:Union{Matrix{<:Union{Float64, Complex{Float64}}}, Vector{<:Union{Float64, Complex{Float64}}}}},
               b::Vector{<:Union{Float64, Complex{Float64}}}; kws...)
               #C::Vector{<:Union{Vector{Float64}, SparseVector{Float64}}}, b::Vector{Float64})
    options = Dict{String, Any}(string(key) => value for (key, value) in kws)
    @assert all(i -> size(At[i], 2) == length(b), 1:length(At))
    @assert length(At) == size(blk, 1)
    #@assert all(i -> size(A[i], 1) == dim(blk[i, 1], blk[i, 2]), 1:length(A))
    #@assert all(i -> length(C[i], 1) == dim(blk[i, 1], blk[i, 2]), 1:length(A))
    # There are 6 output arguments so we use `6` below
    obj, X, y, Z, info, runhist = mxcall(:sdpt3, 6, blk, At, C, b, options)
    return obj, _array.(X), _array(y), _array.(Z), info, runhist
end

include("MOI_wrapper.jl")

end # module
