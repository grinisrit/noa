# Geometric Hamiltonian Monte Carlo (GHMC)

We aim to make it easier to compute_integral Bayesian computation algorithms with larger simulation frameworks, as well as performance demanding systems such as in streaming analytics, games, high frequency trading and many other applications.

This `C++17` library implements Hamiltonian Monte Carlo ([HMC](https://www.sciencedirect.com/science/article/abs/pii/037026938791197X)) schemes over [LibTorch](https://pytorch.org/cppdocs/). The forcus is on high-dimensional problems. 

Currently we have implemented the Explicit [RMHMC](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x) scheme developed by [A.D.Cobb et al.](https://arxiv.org/abs/1910.06243), and initially released in the [hamiltorch](https://github.com/AdamCobb/hamiltorch) package.

In the near future, our research is focused on enhancing this scheme with the [NUTS](https://jmlr.org/papers/v15/hoffman14a.html) algorithm. 
We also plan to provide utilities for parallel execution accross heterogeneous hardware.

:warning: This library needs further numerical testing before release.

## Installation 

Currently, we support only `Linux x86_64` and `CUDA` for GPU (check [WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for Windows). 

This is a header only library, so you can directly drop the `include/ghmc` folder into your project. Otherwise, you can add the repository as a `cmake` submodule. 

The core of the library depends on [LibTorch cxx11 ABI](https://pytorch.org/get-started/locally) (tested with version `1.7.1`). Additionally, you might want to install:
*  [googletest](https://github.com/google/googletest) to run tests.
*  [benchmark](https://github.com/google/benchmark) to run benchmarks.
*  [gflags](https://github.com/gflags/gflags) to run examples.
*  [pugixml](https://github.com/zeux/pugixml) is used by the component `ghmc/pms`.


We strongly encourage you to work within `conda`. The provided environment contains all the required libraries:
```
$ conda env create -f env.yml
$ conda activate ghmc
```
Now, test & install the library (to turn testing off add `-DBUILD_TESTING=OFF`):
```
$ mkdir -p build && cd build
$ cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ cmake --build . --config Release --target install
$ ctest -V
```
In your `CMakeLists.txt` file, make sure you add `LibTorch` and then you can link `GHMC`:
```cmake
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(GHMC CONFIG REQUIRED)
target_link_libraries(your_target torch GHMC::GHMC)
target_compile_options(your_target PRIVATE -Wall -Wextra -Wpedantic -O3)
```

## Usage and Applications

The library offers several advanced applications for Bayesian computation. Documentation and usage examples for each component are provided here:
* [ghmc](apps/sampler) the core component focused on the geometric HMC algorithm dedicated to sample from higher-dimensional probability distributions. The rest of the library builds on top of it.
* [ghmc::pms](apps/pms) provides a framework for simulating the passage of particles through matter.

## Contributions and Support

We welcome contributions to the project and would love to hear about any feature requests. For commercial support or consultancy services you are welcome to contact [GrinisRIT](https://www.grinisrit.com).

(c) 2021 Roland Grinis, GrinisRIT ltd.