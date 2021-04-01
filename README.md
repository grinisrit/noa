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
*  [pugixml](https://github.com/zeux/pugixml) used by the component `ghmc/pms`.
*  [googletest](https://github.com/google/googletest) to run tests.
*  [benchmark](https://github.com/google/benchmark) to run benchmarks.


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

## Usage

The log density function, that we sample from, should be compatible with `torch::autograd`. It must be built out of instances of `torch::autograd::Function` or `torch::nn::Module` richly available from the `PyTorch C++ API`. The user can provide custom extensions if needed, see this [tutorial](https://pytorch.org/tutorials/advanced/cpp_autograd.html).
```cpp
#include <torch/torch.h>
#include <ghmc/ghmc.hh>

int main()
{ 
    // Sample from the 10 dimensional Funnel distribution
    // HMC requires the log density up to additive constants
    auto alog_funnel = [](const auto &w) {
        auto dim = w.numel() - 1;
        return -0.5 * ((torch::exp(w[0]) * w.slice(0, 1, dim + 1).pow(2).sum()) +
                    (w[0].pow(2) / 9) - dim * w[0]);
    };
    auto params_init = torch::ones(11);  
    params_init[0] = 0.;
    auto conf = ghmc::SampleConf{}
                    .set_num_iter(100)
                    .set_leap_steps(25)
                    .set_epsilon(0.14)
                    .set_binding_const(10.);
    auto result = ghmc::sample(alog_funnel,params_init, conf);

    if(result.has_value())
    {   
        // save result for analysis
        torch::Tensor sample = std::get<1>(result.value());;
        torch::save(sample, "sample.pt");
        return 0;
    } else {
        // simulation failed
        return 1;
    }
}
```
After running the above simulation, you may explore the sample in `python`:
```python
import torch
sample = next(torch.jit.load('sample.pt').parameters())
```

It is also possible to rely on [TorchScript models](https://pytorch.org/tutorials/advanced/cpp_export.html) to build the acyclic graph. For instance, one can intially create a net in `pytorch`:
```python
class Net(torch.nn.Module):
    def __init__(self, layer_sizes, bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.bias = bias
        self.l1 = torch.nn.Linear(
            layer_sizes[0], layer_sizes[1], bias=self.bias)
        self.l2 = torch.nn.Linear(
            layer_sizes[1], layer_sizes[2], bias=self.bias)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x.pow(2).sum()

torch.jit.script(Net(layer_sizes=[4, 3, 3])).save('jitmodel.pt')
```
and then reuse it back in the `C++` simulation:
```cpp
// don't forget to include this header
#include <torch/script.h>

// better to run nets on GPU
auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
// load the model 
auto jitmodel = torch::jit::load("jitmodel.pt");
jitmodel.to(device); 
// you can use this net while building the log density
auto net = [&jitmodel](const auto& x){
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(x);
        return jitmodel.forward(inputs).toTensor();
};
```
(c) 2021 Roland Grinis, GrinisRIT ltd.