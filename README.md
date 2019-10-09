# SDPT3

`SDPT3.jl` is an interface to the **[SDPT3](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3/)**
solver. It exports the `sdpt3` function that is a thin wrapper on top of the
`sdpt3` MATLAB function and use it to define the `SDPT3.Optimizer` object that
implements the solver-independent `MathOptInterface` API.

To use it with JuMP, simply do
```julia
using JuMP
using SDPT3
model = Model(with_optimizer(SDPT3.Optimizer))
```
To suppress output, do
```julia
model = Model(with_optimizer(SDPT3.Optimizer, printlevel=0))
```

## Installation

You can install `SDPT3.jl` through the Julia package manager:
```julia
] add https://github.com/JuliaOpt/SDPT3.jl.git
```
but you first need to make sure that you satisfy the requirements of the
[MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) Julia package and that
the SDPT3 software is installed in your
[MATLAB™](http://www.mathworks.com/products/matlab/) installation.
