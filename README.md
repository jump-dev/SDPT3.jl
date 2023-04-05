# SDPT3.jl

[SDPT3.jl](https://github.com/jump-dev/SDPT3.jl) is wrapper for the
[SDPT3](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3) solver.

The wrapper has two components:

 * an exported `sdpt3` function that is a thin wrapper on top of the
   `sdpt3` MATLAB function
 * an interface to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)

## Affiliation

This wrapper is maintained by the JuMP community and is not an official wrapper
of SDPT3.

## License

`SDPT3.jl` is licensed under the [MIT License](https://github.com/jump-dev/v.jl/blob/master/LICENSE.md).

The underlying solver, [SDPT3](https://blog.nus.edu.sg/mattohkc/softwares/sdpt3/)
is licensed under the [GPL v2 License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

In addition, SDPT3 requires an installation of MATLAB, which is a closed-source
commercial product for which you must [obtain a license](https://www.mathworks.com/products/matlab.html).

## Use with JuMP

To use SDPT3 with [JuMP](https://github.com/jump-dev/JuMP.jl), do:
```julia
using JuMP, SDPT3
model = Model(SDPT3.Optimizer)
set_attribute(model, "printlevel", 0)
```

## Installation

First, make sure that you satisfy the requirements of the
[MATLAB.jl](https://github.com/JuliaInterop/MATLAB.jl) Julia package, and that
the SeDuMi software is installed in your
[MATLAB™](http://www.mathworks.com/products/matlab/) installation.

Then, install `SDPT3.jl` using `Pkg.add`:
```julia
import Pkg
Pkg.add("SDPT3")
```

### SDPT3 not in PATH

If you get the error:
```
Error using save
Variable 'jx_sdpt3_arg_out_1' not found.

ERROR: LoadError: MATLAB.MEngineError("failed to get variable jx_sdpt3_arg_out_1 from MATLAB session")
Stacktrace:
[...]
```
The error means that we could not find the `sdpt3` function with one output
argument using the MATLAB C API. This most likely means that you did not add
SDPT3 to the MATLAB's path (that is, the `toolbox/local/pathdef.m` file).

If modifying `toolbox/local/pathdef.m` does not work, the following should work,
where `/path/to/sdpt3/` is the directory where the `sdpt3` folder is located:
```julia
julia> using MATLAB

julia> cd("/path/to/sdpt3/") do
           MATLAB.mat"install_sdpt3"
       end

julia> MATLAB.mat"savepath"
```

An alternative fix is suggested in the [following issue](https://github.com/jump-dev/SDPT3.jl/issues/9#issuecomment-855509257).

### Error in validate

If you get the error:
```
Brace indexing is not supported for variables of this type.

Error in validate

Error in sdpt3 (line 171)
   [blk,At,C,b,blkdim,numblk,parbarrier] = validate(blk,At,C,b,par,parbarrier);

Error using save
Variable 'jx_sdpt3_arg_out_1' not found.
```
It might mean that you have added [SDPNAL](https://github.com/jump-dev/SDPNAL.jl)
in addition to SDPT3 in the MATLAB's path (that is, the `toolbox/local/pathdef.m`
file). Because SDPNAL also defines a `validate` function, this can make `sdpt3`
call SDPNAL's `validate` function instead of SDPT3's `validate` function, which
causes the issue.

One way to fix this from the Julia REPL is to reset the search path to the
factory-installed state using `restoredefaultpath`:
```julia
julia> using MATLAB

julia> MATLAB.restoredefaultpath()

julia> MATLAB.mat"savepath"
```
