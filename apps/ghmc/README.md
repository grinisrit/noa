# Geometric Hamiltonian Monte-Carlo (GHMC)

To build theses examples add `-DBUILD_GHMC_APPS=ON` to the top level cmake command.

The library implements Hamiltonian Monte Carlo ([HMC](https://www.sciencedirect.com/science/article/abs/pii/037026938791197X)) schemes over [LibTorch](https://pytorch.org/cppdocs/). The forcus is on high-dimensional problems. 

Currently we have implemented the Explicit [RMHMC](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x) scheme developed by [A.D.Cobb et al.](https://arxiv.org/abs/1910.06243), and initially released in the [hamiltorch](https://github.com/AdamCobb/hamiltorch) package.

In the near future, our research is focused on enhancing this scheme with the [NUTS](https://jmlr.org/papers/v15/hoffman14a.html) algorithm. 
We also plan to provide utilities for parallel execution accross heterogeneous hardware.

## Usage 

The log density function, that we sample from, should be compatible with `torch::autograd`. It must be built out of instances of `torch::autograd::Function` or `torch::nn::Module` richly available from the `PyTorch C++ API`. The user can provide custom extensions if needed, see this [tutorial](https://pytorch.org/tutorials/advanced/cpp_autograd.html).
```cpp
#include <torch/torch.h>
#include <kotik/ghmc.hh>

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