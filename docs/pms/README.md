# Passage of Particles through Matter Simulations (PMS) 

:warning: This component is under active development.

Basic usage examples can be found in [functional tests](../../test/pms).

The materials configuration files for the physical models are available from [noa-pms-models](https://github.com/grinisrit/noa-pms-models).

To enable parallel execution for some algorithms you should link against `OpenMP`.

To load MDF configurations we rely on [pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

To build `CUDA` routines specify at `cmake` command line `-DBUILD_NOA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` (or the GPU architecture of your choice). 
Enabling separable compilation, you should add the kernels to a `.cu` source file:
```cpp
// in kernels.cu 
#include <noa/pms/kernels.cuh>
```

(c) 2021 Roland Grinis, GrinisRIT ltd.