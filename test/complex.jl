using Test
using SDPT3

@testset "Project to hermitian PSD" begin
    blk = ["q" 4.0
           "s" 2.0]
    C = Union{Matrix{Float64}, Vector{Float64}}[[1.0, 0.0, 0.0, 0.0], zeros(2, 2)]
    A = Matrix{Float64}[
         [0.0 1.0 0.0 0.0
          0.0 0.0 1.0 0.0
          0.0 0.0 0.0 1.0]',
         [1.0 0.0 0.0
          0.0 √2 0.0
          0.0 0.0 1.0]']
    b = [1.0, -√2 + √2 * im, -1.0]
#    blk = ["s" 2.0]
#    c = [0.0 1.0
#         1.0 0.0]
#    A = [1.0 0.0 0.0
#         0.0 0.0 1.0]
#    b = [1.0
#         1.0]
    obj, X, y, Z, info, runhist = sdpt3(blk, A, C, b)
#obj, X, y, Z, info, runhist = sdpt3(blk, [Matrix(A')], C, b)
    @show obj
    @show X
    @show y
    @show Z
    @show info
    @show runhist
end
