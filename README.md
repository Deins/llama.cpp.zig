# llama.cpp.zig
llama.cpp bindings and utilities for zig

* Implements llama.h for nicer interaction with zig. Removes prefixes. Groups functions within Context/Model etc structs. camelCaseFunctionName etc. Depends on translate-c as I tried to not rewrite struct/enum definitions too much as those might change, but some were written Batch, TokenType - has to be seen if more of them might benefit implementing.
* Re-Implements some of the C++ that is not acessable through c api due to use of c++ std stuff:
    * [sampling zig](./src/ sampling zig)
* Implements some utilities such as Tokenizer, Detokenizer structs for ease of use.

## Usage
Clone: `git clone --recursive https://github.com/Deins/llama.cpp.zig.git`  
Build and run [example](examples/simple.zig):
Download llama.cpp supported model, usually in .gguf format. For example [this one](https://huggingface.co/TheBloke/rocket-3B-GGUF).
Build & run:
```bash
zig build run-simple -Doptimize=ReleaseFast -- --model_path /path_to/model.gguf --prompt "Hello! I am AI, and here are the 10 things I like to think about:"
```



## TODO:
* finish & cleanup simple sample, try more advanced scenarios
* `llama.zig` has last few more functions at end of the file that need to be added.
* integration with package manager zig.zon etc. Overall clean up build.zig, push needed changes for nice integration upstrem
* figure out if we can implement compiling some of the BLAS libraries for gpu acceleration from zig, currently has cpu base implementation only
* .... 
