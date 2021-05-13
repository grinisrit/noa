# Geometric Hamiltonian Monte-Carlo (GHMC)

The library implements Hamiltonian Monte Carlo ([HMC](https://www.sciencedirect.com/science/article/abs/pii/037026938791197X)) schemes over [LibTorch](https://pytorch.org/cppdocs/). The forcus is on high-dimensional problems. 

Currently we have implemented the Explicit [RMHMC](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x) scheme developed by [A.D.Cobb et al.](https://arxiv.org/abs/1910.06243), and initially released in the [hamiltorch](https://github.com/AdamCobb/hamiltorch) package.

In the near future, our research is focused on enhancing this scheme with the [NUTS](https://jmlr.org/papers/v15/hoffman14a.html) type algorithms. 

## Usage 

Basic usage examples can be found in [functional tests](../../test/ghmc).

The log density function, that we sample from, should be compatible with `torch::autograd`. 
It must be built out of instances of `torch::autograd::Function` or `torch::nn::Module` richly available from the `PyTorch C++ API`. 
The user can provide custom extensions if needed, see this [tutorial](https://pytorch.org/tutorials/advanced/cpp_autograd.html).
It is also possible to rely on [TorchScript models](https://pytorch.org/tutorials/advanced/cpp_export.html), 
which is in fact the recommended way to work with deep learning models. 
For instance, one can initially create a net in `pytorch`:
```python
import torch

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
Use it back in the `C++` simulation:
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