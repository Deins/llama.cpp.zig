# llama.cpp.zig
llama.cpp bindings and utilities for zig. Currently targeting zig `0.11.x`.

* Implements llama.h for nicer interaction with zig.
    * Removes prefixes, changes naming for functions to camelCase. Group functions within most appropriete struct. etc.
    * Bindings still depend on translate-c as I tried to not rewrite struct definitions too much as those might change. But some were rewritten for ease of use. Has to be seen if others might benefit rewriting as well for nicer access / syntax.
* Re-Implements some of the C++ that is not acessable through c api due to use of c++ std stuff:
    * [sampling.zig](./llama.cpp.zig/sampling.zig)
* Implements some utilities such as:
    * buffered Tokenizer
    * buffered Detokenizer
    * basic templated prompt generator, to try to manage different prompt formatting styles (chatML, alpaca, etc). 

## Example usage
Clone: `git clone --recursive https://github.com/Deins/llama.cpp.zig.git`
1. Download llama.cpp supported model (usually *.gguf format). For example [this one](https://huggingface.co/TheBloke/rocket-3B-GGUF).
2. build and run with:
```bash
# set the path to model
zig build run-simple -Doptimize=ReleaseFast -- --model_path path_to/model.gguf --prompt "Hello! I am AI, and here are the 10 things I like to think about:"
```
See [examples/simple.zig](examples/simple.zig) 


## CLBlast acceleration
Clblast & opencl is supported by building it from source with zig. Cuda backend is not finished as I don't have nvidia hardware, pull requests welcome.

### Usage:
Ideally just `zig build -Dclblast ...`. It should work out of the box if you have installed [GPUOpen/ocl](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases). 
For other configurations you will need to find where OpenCL headers/libs are and pass them in using arguments `zig build -Dclblast -Dopencl_includes="/my/path" -Dopencl_libs="/my/path/"`
Auto detection might be improved in future - let me know what opencl sdk you use. 

