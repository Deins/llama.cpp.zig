# llama.cpp.zig
llama.cpp bindings and utilities for zig. Currently targeting zig `0.13.x`. 

* Provides build from source using zig build.
* Implements llama.h for nicer interaction with zig.
    * Removes prefixes, changes naming for functions to camelCase. Groups functions within most appropriete struct. etc.
    * Bindings partially depend on translate-c partially rewritten for ease of use.
* (Deprecated, llama.cpp api now has this)~~Re-Implements some of the C++ that is not acessable through c api due to use of c++ std stuff:  [sampling.zig](./llama.cpp.zig/sampling.zig)~~

* Implements some utilities such as:
    * buffered Tokenizer & Detokenizer
    * (deprecated due to api changes, need to be revisited) ~~prompt utility to simplify interaction with llama. (still wip) will support easly modifying prompt & regenerating it. Handling of out of context behaviour, etc. Possibly optional pagination/indexing of messages.~~
    * (llama.cpp now has this, zig bindings need to be revisited)  ~~basic templated prompt generator, to try to manage different prompt formatting styles (chatML, alpaca, etc).~~

Due to not keeping up with incremental llama.cpp api changes, not all bindings are in sync at the moment.  Use translate-c api directly for anything thats missing. 

## Example usage
Clone: `git clone --recursive https://github.com/Deins/llama.cpp.zig.git`
1. Download llama.cpp supported model (usually *.gguf format). For example [this one](https://huggingface.co/TheBloke/rocket-3B-GGUF).
2. build and run with:
```bash
zig build run-simple -Doptimize=ReleaseFast -- --model_path path_to/model.gguf --prompt "Hello! I am LLM, and here are the 5 things I like to think about:"
```
See [examples/simple.zig](examples/simple.zig) 

### CPP samples
Subset of llama cpp samples have been included in build scripts. Use `-Dcpp_samples` option to install them.  
Or run them directly, for example: `zig build run-cpp-main -Doptimize=ReleaseFast -- -m path/to/model.gguf -p "hello my name is"`

## Tested platforms
* ☑️ x86_64 windows
* ☑️ x86_64 linux (WSL Ubuntu 22)

## Backend support
| Backend       | Support       | Comment       |
| ------------- | ------------- | ------------- |
| cpu           | ☑️           | |
| cuda          | | 
| metal         | | 
| sycl          | | 
| vulkan        | | 
| opencl        | | 
| cann          | | 
| blas          | | 
| rpc           | | 
| kompute       | | 
| CLBlast       | ❌ | deprecated, was supported in older: [8798dea](https://github.com/Deins/llama.cpp.zig/commit/8798dea5fcc62490bd31bfc36576db93191b7e43) |

