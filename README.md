# NOA: Bayesian Computation for Deep Learning

We aim to make it easier to integrate Bayesian computation algorithms with Deep Learning applications, larger simulation frameworks, as well as performance demanding systems such as ones encountered in streaming analytics, games, high frequency trading and many other applications.

## Installation 

Currently, we support only `GNU` and `CUDA` for GPU (check [WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for Windows). 

The core of the library depends on [LibTorch (cxx11 ABI)](https://pytorch.org/get-started/locally) tested with version `1.7.1`. For additional configuration needed by some applications please refer to the documentation [below](#applications).

We encourage you to work with `conda`. The provided environment contains all the required libraries:
```
$ conda env create -f env.yml
$ conda activate noa
```
Build, test & install the library (to turn testing off add `-DBUILD_NOA_TESTS=OFF`):
```
$ mkdir -p build && cd build
$ cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ cmake --build . --config Release --target install
$ ctest -V
```
To build benchmarks specify `-DBUILD_NOA_BENCHMARKS=ON`.

In your `CMakeLists.txt` file, make sure you add `LibTorch` and then you can link `noa`:
```cmake
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(NOA CONFIG REQUIRED)
target_link_libraries(your_target torch noa)
target_compile_options(your_target PRIVATE -Wall -Wextra -Wpedantic -O3)
```

## Applications

NOA offers several advanced applications for Bayesian computation. Please refer to the documentation and usage examples for each component to find out more:
* [GHMC](apps/ghmc) the core component focused on the Geometric HMC algorithm dedicated to sampling from higher-dimensional probability distributions. The rest of the library builds on top of it.
* [PMS](apps/pms) provides a framework for simulating the passage of particles through matter. 

## Contributions and Support

We welcome contributions to the project and would love to hear about any feature requests. For commercial support or consultancy services you are welcome to contact [GrinisRIT](https://www.grinisrit.com).

(c) 2021 Roland Grinis, GrinisRIT ltd.