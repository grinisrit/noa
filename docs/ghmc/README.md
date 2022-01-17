# Geometric Hamiltonian Monte-Carlo (GHMC)

The library implements Hamiltonian Monte-Carlo 
([HMC](https://www.sciencedirect.com/science/article/abs/pii/037026938791197X)) 
schemes over [LibTorch](https://pytorch.org/cppdocs/). 
The focus is on high-dimensional problems. 

Currently, we have implemented the explicit 
[RMHMC](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00765.x) 
scheme developed by [A.D.Cobb et al.](https://arxiv.org/abs/1910.06243). 
A standard HMC algorithm is also available.

In the near future, our research focus on enhancing this scheme with 
[NUTS](https://jmlr.org/papers/v15/hoffman14a.html) type algorithms. 

## Usage 
 
In the [notebook](bayesian_deep_learning.ipynb) you will find an introduction to Geometric Hamiltonian Monte-Carlo and a tutorial on Bayesian Deep Learning.

A [talk](https://www.youtube.com/watch?v=d6ezzxzqEaA&t=25s) covering
those topics has been given at 
[itCppCon21](https://italiancpp.org/itcppcon21) conference.

Basic usage examples can be found in [functional tests](../../test/ghmc). The log density function, that we sample from, should be compatible with `torch::autograd`. 
It must be built out of instances of `torch::autograd::Function` or `torch::nn::Module` richly available from the PyTorch `C++` API. 

The user can provide custom extensions if needed, 
see [tutorial](https://pytorch.org/tutorials/advanced/cpp_autograd.html).
It is also possible to rely on 
[TorchScript](https://pytorch.org/tutorials/advanced/cpp_export.html), 
which is in fact the recommended way to work with deep learning models. 

:warning: The library needs further numerical testing before release. 

## Acknowledgements

Implementations of the HMC algorithms above over [PyTorch](https://pytorch.org) 
are also available in the 
[hamiltorch](https://github.com/AdamCobb/hamiltorch) package.

(c) 2022 GrinisRIT ltd. and contributors
