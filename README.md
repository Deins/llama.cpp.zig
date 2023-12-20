# llama.cpp.zig
llama.cpp bindings and utilities for zig. Currently targeting zig `0.11.x`.

* Implements llama.h for nicer interaction with zig. Such as:
    * Removes prefixes, changes naming for functions to camelCase.
    * Groups functions within most appropriete struct. 
    * Still depends on translate-c as I tried to not rewrite struct/enum definitions too much as those might change, but some were rewritten. Has to be seen if others might benefit rewriting as well for nicer access / syntax.
* Re-Implements some of the C++ that is not acessable through c api due to use of c++ std stuff:
    * [sampling.zig](./llama.cpp.zig/sampling.zig)
* Implements some utilities such as Tokenizer, Detokenizer structs for ease of use. WIP, basic templated prompt generator (mainly to be able to easaly run same promts on different propt formats chatML/alpaca etc.)

## Usage
Clone: `git clone --recursive https://github.com/Deins/llama.cpp.zig.git`
1. Download llama.cpp supported model, usually in .gguf format. For example [this one](https://huggingface.co/TheBloke/rocket-3B-GGUF).
2. build and run with:
```bash
# change the path to model
zig build run-simple -Doptimize=ReleaseFast -- --model_path /path_to/model.gguf --prompt "Hello! I am AI, and here are the 10 things I like to think about:"
```
See [example](examples/simple.zig) 



### todo
* finish & cleanup simple sample, try more advanced scenarios
* `llama.zig` has last few more functions at end of the file that need to be added.
* integration with package manager zig.zon etc. Overall clean up build.zig, push needed changes for nice integration upstrem
* figure out if we can implement compiling some of the BLAS libraries for gpu acceleration from zig, currently has cpu base implementation only
* .... 
