# NOA: Nonlinear Optimisation Algorithms

We aim to make it easier to integrate optimisation with on-line learning, bayesian computation and 
large simulation frameworks through dedicated differentiable algorithms. 
Our solution is suitable for both research 
and applications in performance demanding systems 
such as encountered in streaming analytics, game development 
and high frequency trading.

## Installation 

Currently, we support only `GNU`, and `CUDA` for GPU 
(check [WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for Windows,
and [LIMA](https://github.com/lima-vm/lima) for macOS).
A toolchain fully supporting `C++17` is required.
`NOA` is a header-only library, so you can directly 
drop the `src/noa` folder into your project.

:warning: However, beyond `C++17` source code, the project contains:
* `CUDA` sources in `noa/kernels.cuh`
* `C++` third-party sources in `noa/kernels.hh`

which may require separable compilation.

#### Note 1
`src/noa/kernels.hh` could be compiled separately via `test/kernels.cc`, but it is also possible to include it directly into your program if you want to keep everything inside of the single trasnlation unit.
#### Note 2: PUMAS
PUMAS code under `src/noa/3rdparty/_pumas` was syntactically adapted to compile under `g++` with `-fpermissive`.
To avoid setting this parameter everywhere, `noa/kernels.hh` doesn't include PUMAS by defaut.
To include PUMAS, define `NOA_3RDPARTY_PUMAS` before including `noa/kernels.hh`, and compile the corresponding file with an `-fpermissive` flag.

### CMake project
The core of the library depends on 
[LibTorch](https://pytorch.org/get-started/locally) 
(which is also distributed via `pip` and `conda`).

You can specify whether torch was compiled with C++11 ABI 
setting `TORCH_USE_CXX11_ABI` to 0 or 1. 
If we manage to fetch the value of `_GLIBCXX_USE_CXX11_ABI`
during configuration, your choice will be overwritten.
By default, this variable is 0.


For additional configuration needed by some applications
please refer to the documentation [below](#applications).

We encourage you to work with `conda`. 
If your system supports `CUDA`, the environment [env.yml](env.yml) 
contains all the required libraries:
```
$ conda env create -f env.yml
$ conda activate noa
```
For a `CPU` only installation please use [env-cpu.yml](env-cpu.yml) instead.

Build tests & install the library 
(to turn testing off add `-DBUILD_NOA_TESTS=OFF`):
```
$ mkdir -p build && cd build
$ cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ cmake --build . --target install
$ ctest -V
```
To build benchmarks specify `-DBUILD_NOA_BENCHMARKS=ON`. 
To enable parallel execution for some algorithms you should link against `OpenMP`.
To build `CUDA` tests add `-DBUILD_NOA_CUDA=ON` 
and the  GPU architecture of your choice,
e.g. `-DCMAKE_CUDA_ARCHITECTURES=75`.

To build documentation you need to have [doxygen](https://doxygen.nl/) installed and 
specify `-DBUILD_DOCS=ON`.

Finally, once `NOA` is installed, 
you can link against it in your own `CMakeLists.txt` file.
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
To build the [JNI wrapper](jnoa) you need to add `-DBUILD_JNOA=ON` setting `-DJAVA_HOME=<...>` if necessary. 
This will produce the shared library `jnoa` to which
you should point the `java.library.path` for the `JVM` to load it.

## Applications

`NOA` offers several advanced applications for optimisation. 
Please refer to the documentation and usage examples 
for each component to find out more:
* [GHMC](docs/ghmc) focuses on Bayesian computation 
with the Geometric HMC algorithms dedicated to sampling 
from higher-dimensional probability distributions. 
* [PMS](docs/pms) provides a framework for solving inverse problems
in the passage of particles through matter simulations. 
* [CFD](docs/cfd) implements adjoint sensitivity models for a variety 
problems arising in computational fluid dynamics.
* [QC](docs/qc) is a differentiable quantum chemistry platform for materials design.
* [QUANT](docs/quant) a differentiable derivative 
pricing library.

## Acknowledgments

When referring to this library in scientific work please cite

```
@article{grinis2022diffprog,
  title={Differentiable programming for particle physics simulations},
  author={Roland Grinis},
  journal={JETP Vol. 161 (2)},
  year={2022}
}
```

We welcome contributions to the project 
and would love to hear about any feature requests.

The JNI wrapper is being developed in collaboration with 
[KMath contributors](https://github.com/mipt-npm/kmath/graphs/contributors).

## License

NOA is licensed under the terms and conditions of the GNU General
Public License (GPL) version 3 or - at your option - any later
version. The GPL can be read [online](https://www.gnu.org/licenses/gpl-3.0.en.html), 
or see the full [LICENSE](LICENSE)

Please note that NOA license does NOT feature a
template exception to the GNU General Public License. This means that
you must publish any source code which uses any of the NOA header
files if you want to redistribute your program to third parties. If
this is unacceptable to you, please [contact us](info@grinisrit.com) 
for a commercial license.

For support or consultancy services 
contact [GrinisRIT](https://www.grinisrit.com).

(c) 2022 GrinisRIT ltd. 
