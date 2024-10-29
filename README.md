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
`src/noa/kernels.hh` could be compiled separately via `test/kernels.cc`, but it is also possible to include it directly into your program if you want to keep everything inside of the single translation unit.
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

### Contributors

* [Sabina Abdiganieva](https://github.com/abdiganieva) researched various DFT implementations 
for QC.

* [Dmitry Bazanov](https://github.com/zugzvangg) developed the Heston volatility surface calibration
routines in QUANT and works on its applications to real market data. 

* [Gregory Dushkin](https://github.com/GregTheMadMonk) is the main developer for CFD, implementing 
differentiable programming algorithms for MHFEM schemes and integrating TNL into the library. 
He also works on muon simulations in PMS and intergrated pumas.
He helped to implement the adjoint sensitivity algorithm for Kohn-Sham equations in QC.

* [Anastasia Golovina](https://github.com/a-tass) wrapped LibTorch's optimizers
into KMath.

* [Eugen Kadilenko](https://github.com/ruthenium96) developed the adjoint sensitivity
model for Kohn-Sham equations in QC.

* [Andrey Lipsky](https://github.com/Molozey) works on applications to real
market data for QUANT.

* [Oleg Mosyagin](https://github.com/F5DXWsqPme) implemented the XPU ray-tracing
algorithm in PMS.

* [Ivan Novikov](https://github.com/mrFendel) contributes to the documentation
in QUANT, implemented the PDE methods for the Black-Scholes 
model and works on margin models.

* [Danila Ryazanov](https://github.com/DanilaSR) developed the XPU DCS 
computations for muons in PMS.

* [Maksim Sosnin](https://github.com/mmssss) implemented the almost exact simulation
scheme for the Heston Model in QUANT and works on integration with Deep Hedging.

* The JNI wrapper is being developed in collaboration with 
[KMath contributors](https://github.com/mipt-npm/kmath/graphs/contributors).

We welcome contributions to the project 
and would love to hear about any feature requests

## License

NOA is licensed under the terms and conditions of the GNU General
Public License (GPL) version 3 or - at your option - any later
version. The GPL can be read [online](https://www.gnu.org/licenses/gpl-3.0.en.html), 
or see the full [LICENSE](LICENSE).

Please note that NOA license does NOT feature a
template exception to the GNU General Public License. This means that
you must publish any source code which uses any of the NOA header
files if you want to redistribute your program to third parties. If
this is unacceptable to you, please [contact us](info@grinisrit.com) 
for a commercial license.

For support or consultancy services 
contact [GrinisRIT](https://www.grinisrit.com).

(c) 2023 GrinisRIT ltd. and contributors
