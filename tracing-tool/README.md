## Tracing Tool

The ***tracing-tool*** is used to extract the memory and compute traces. This tool uses and extends NVBit (NVidia Binary Instrumentation Tool) which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. Licence and agreement of NVBIT is found in the origianal [NVBIT repo](https://github.com/NVlabs/NVBit) ("This software contains source code provided by NVIDIA Corporation")

NVBIT does not require application source code, any pre-compiled GPU application should work regardless of which compiler (or version) has been used (i.e. nvcc, pgicc, etc).

## Usage

*  Setup the `MAX_KERNELS` variable in `tracer.cu` to define the limit on the number of kernels you want to instrument in the application. The `MAX_KERNELS` variable we used for collecting traces is 300.

* For stanalone building and running of the ***tracing-tool***, please see below: 

  #### 1. Building the tool
  
  * Setup `ARCH` and `NVCC` variable in the Makefile. For the Volta architecture, you need to set:
    ```shell
    NVCC=/usr/local/cuda/bin/nvcc -ccbin=$(CXX) -D_FORCE_INLINES --compiler-options "-pipe"
    ARCH=70
    ```
    It is important to note that this tool is not sensitive to CUDA versions, so your default version should be fine.
  * Compile the ***tracing-tool***:
    ```
    make clean && make
    ```

  #### 2. Extracting the traces
  
  ```
  LD_PRELOAD=/path/to/tracing-tool/tracer.so /path/to/app [parameters of app] 
  ```
  
  The above command outputs two folders ***memory_traces*** and ***sass_traces*** each has the applications kernel traces. It also output ***configs*** file which has information about the kernel executing inside the application. 
