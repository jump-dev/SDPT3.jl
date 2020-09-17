# SDPT3

`SDPT3.jl` is an interface to the **[SDPT3](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3/)**
solver. It exports the `sdpt3` function that is a thin wrapper on top of the
`sdpt3` MATLAB function and use it to define the `SDPT3.Optimizer` object that
implements the solver-independent `MathOptInterface` API.

To use it with JuMP, simply do
```julia
using JuMP
using SDPT3
model = Model(SDPT3.Optimizer)
```
To suppress output, do
```julia
model = Model(optimizer_with_attributes(SDPT3.Optimizer, printlevel=0))
```

## Installation

You can install `SDPT3.jl` through the Julia package manager:
```julia
] add SDPT3
```
but you first need to make sure that you satisfy the requirements of the
[MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) Julia package and that
the SDPT3 software is installed in your
[MATLAB™](http://www.mathworks.com/products/matlab/) installation.

### Troubleshooting

If you get the error:
```
Brace indexing is not supported for variables of this type.

Error in validate

Error in sdpt3 (line 171)
   [blk,At,C,b,blkdim,numblk,parbarrier] = validate(blk,At,C,b,par,parbarrier);

Error using save
Variable 'jx_sdpt3_arg_out_1' not found.
```
It might means that you have added [SDPNAL](https://github.com/jump-dev/SDPNAL.jl) in addition to SDPT3 in the MATLAB's path (i.e. the `toolbox/local/pathdef.m` file).
As SDPNAL also define a `validate` function, this makes `sdpt3` calls SDPNAL's `validate` function instead of SDPT3's `validate` function which causes the issue.
