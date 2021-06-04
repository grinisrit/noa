# Passage of Particles through Matter Simulations (PMS) 

:warning: This component is under active development.

An introduction for differentiable programming for particle physics simulations is presented in the [notebook](differentiable_programming_pms.ipynb). Basic usage examples can be found in [functional tests](../../test/pms).

Physics model configuration files are available from [noa-pms-models](https://github.com/grinisrit/noa-pms-models). The current focus is on Muons and Taus, but we plan to cover a wider range of particles. To load the MDF settigns we rely on [pugixml](https://github.com/zeux/pugixml)  version `1.11` provided.

To enable parallel execution for some algorithms you should link against `OpenMP`.
To build `CUDA` routines specify `-DBUILD_NOA_CUDA=ON` and  `-DCMAKE_CUDA_ARCHITECTURES=75` (or the GPU architecture of your choice) to `cmake`. 
Enabling separable compilation, you should add the kernels to a `.cu` source file:
```cpp
// in kernels.cu 
#include <noa/pms/kernels.cuh>
```

(c) 2021 Roland Grinis, GrinisRIT ltd.