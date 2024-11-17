# NOA Documentation

In this folder you will find supplementary material for  
our academic work based on `NOA`. Moreover, further examples
and usage cases are discussed, for each component:

* [GHMC](ghmc/README.md) focuses on Bayesian computation
  with the Geometric HMC algorithms dedicated to sampling
  from higher-dimensional probability distributions.
* [PMS](pms/README.md) provides a framework for solving inverse problems
  in the passage of particles through matter simulations.
* [CFD](cfd/README.md) implements adjoint sensitivity models for a variety
  problems arising in computational fluid dynamics.
* [QUANT](quant/README.md) a differentiable derivative
  pricing library. 

## Notebooks set-up

Many examples are provided as `python` jupyter notebooks. 

We provide a `conda` environment [docs.yml](../docs.yml) 
containing all the required dependencies:

```
$ conda env create -f docs.yml
$ conda activate noa-docs
```

If you want to run them on
[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)
or [Datalore](https://datalore.jetbrains.com/notebooks)
clone `NOA`,  executing inside your notebook:

```python
!git clone https://github.com/grinisrit/noa.git
noa_location = 'noa'
```

Also, make sure that `ninja` and `g++-9` or higher are available. The following commands will do that for you:
```python
!pip install Ninja
!add-apt-repository ppa:ubuntu-toolchain-r/test -y
!apt update
!apt upgrade -y
!apt install gcc-9 g++-9
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

Finally, for `GPU` development install `CUDA` 11.2 or higher: 
```python
!sudo apt-get update && sudo apt-get install cuda-nvcc-11-2 -y
```
(c) 2024 GrinisRIT ltd. 