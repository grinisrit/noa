# NOA: Nonlinear Optimisation Algorithms

We aim to make it easier to integrate nonlinear optimisation
algorithms with on-line learning, bayesian computation and large simulation frameworks. 
Our solution is suitable for both research and applications in performance demanding systems such as encountered in streaming analytics, game development and high frequency trading.

## Installation 

Currently, we support only `GNU`, and `CUDA` for GPU (check [WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for Windows).
A toolchain fully supporting `C++17` is required.
`NOA` is a header-only library, so you can directly drop the `include/noa` folder into your project.

### CMake project
The core of the library depends on [LibTorch Pre-cxx11 ABI](https://pytorch.org/get-started/locally) 
(which is also distributed via `pip` and `conda`) tested with version `1.9.0`. 
For additional configuration needed by some applications please refer to the documentation [below](#applications).

We encourage you to work with `conda`. If your system supports `CUDA`, the environment [env.yml](env.yml) contains all the required libraries:
```
$ conda env create -f env.yml
$ conda activate noa
```
For a `CPU` only installation please use [env-cpu.yml](env-cpu.yml) instead.

Build tests & install the library (to turn testing off add `-DBUILD_NOA_TESTS=OFF`):
```
$ mkdir -p build && cd build
$ cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ cmake --build . --target install
$ ctest -V
```
To build benchmarks specify `-DBUILD_NOA_BENCHMARKS=ON`. To enable parallel execution for some algorithms you should link against `OpenMP`.  To build `CUDA` tests add `-DBUILD_NOA_CUDA=ON` and  `-DCMAKE_CUDA_ARCHITECTURES=75` (or the GPU architecture of your choice).

Finally, once `NOA` is installed, you can link against it in your own `CMakeLists.txt` file.
Make sure to add `LibTorch` as well:
```cmake
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(NOA CONFIG REQUIRED)
target_link_libraries(your_target torch NOA::NOA)
target_compile_options(your_target PRIVATE -Wall -Wextra -Wpedantic -O3)
```

### KMath wrapper
`NOA` is exposed within the `kotlin` library
[KMath](https://github.com/mipt-npm/kmath) as a dedicated module
[kmath-noa](https://github.com/mipt-npm/kmath/tree/feature/noa/kmath-noa).
To build the [JNI wrapper](jnoa) you need to add `-DBUILD_JNOA=ON`. 
This will produce the shared library `jnoa` to which
you should point the `java.library.path` for the `JVM` to load it.

## Applications

`NOA` offers several advanced applications for optimisation. Please refer to the documentation and usage examples for each component to find out more:
* [GHMC](docs/ghmc) focus on Bayesian optimisation with the Geometric HMC algorithm 
dedicated to sampling from higher-dimensional probability distributions. 
* [PMS](docs/pms) provides a framework for solving inverse problems
in the passage of particles through matter simulations. 

## Contributions and Support

We welcome contributions to the project and would love to hear about any feature requests.
For commercial support or consultancy services contact [GrinisRIT](https://www.grinisrit.com).

(c) 2021 Roland Grinis, GrinisRIT ltd.
