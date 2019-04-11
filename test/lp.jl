using Test
using SDPT3

# Example page 4 of SeDuMi_Guide_105R5.pdf
@testset "Linear Programming example" begin
    blk = ["l"  4]
    c = [ 1.0
         -1.0
          0.0
          0.0]
    A = [10.0 -7.0 -1.0 0.0
          1.0  0.5  0.0 1.0]
    b = [5.0
         3.0]
    obj, X, y, Z, info, runhist = sdpt3(blk, [Matrix(A')], [c], b)
    tol = 1e-5
    @test obj[1] ≈ -1/8 atol=tol rtol=tol
    @test obj[2] ≈ -1/8 atol=tol rtol=tol
    @test X[1] ≈ [47/24, 25//12, 0, 0] atol=tol rtol=tol
    @test y ≈ [1/8, -1/4] atol=tol rtol=tol
    @test Z[1] ≈ [0, 0, 1/8, 1/4] atol=tol rtol=tol
    @test info["termcode"] == 0.0
end
