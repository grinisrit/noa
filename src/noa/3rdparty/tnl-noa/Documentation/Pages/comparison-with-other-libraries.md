# Comparison with other libraries

## Memory space and execution model

TNL has separate concepts for the memory space and execution model, which are
represented by different template parameters. See the \ref core_concepts
"Core concepts" page for details.

- Most other libraries have separate types for CPU and GPU data structures
  (e.g. `Vector` and `cuVector`):
  - [Thrust](https://github.com/thrust/thrust/): `host_vector`, `device_vector`
    (plus macro-based selection, see below)
  - [Paralution](http://www.paralution.com/documentation/): `HostVector`, `AcceleratorVector`
  - [Bandicoot](https://coot.sourceforge.io/), [Kaldi](http://kaldi-asr.org/doc/about.html)
- These libraries have the concept of a "memory space" which is configurable as
  a template parameter:
  - [CUV](https://github.com/deeplearningais/CUV)
  - [CUSP](http://cusplibrary.github.io/classcusp_1_1array1d.html) - but CUSP
    uses Thrust, so `device_memory` might be the same as `host_memory` if OpenMP
    is used as the `device`
  - [Kokkos](https://github.com/kokkos/kokkos) - they have a concept of a
    "memory space" and "execution space", but there is also some default choice
    of the spaces, possibly even through command-line arguments (in which case
    the array type would be polymorphic, because something has to store the
    current memory/execution space)
- These libraries have transparent access to the data from GPU and CPU:
  - the CUDA toolkit itself, via `cudaMallocManaged`
  - [cudarrays](https://github.com/cudarrays/cudarrays) - they have custom
    virtual memory system using `cudaMalloc` and the standard host allocator
- These libraries select the (default) device based on some macro
  (this approach is way too simple, because multiple different devices cannot be
  combined):
  - Thrust: see [device backends](https://github.com/thrust/thrust/wiki/Device-Backends)
    and [host backends](https://github.com/thrust/thrust/wiki/Host-Backends)
- These libraries do not abstract memory space, only execution model:
  - [RAJA](https://github.com/LLNL/RAJA)
  - [Nebo](https://www.sciencedirect.com/science/article/pii/S0164121216000182)
    (also with a macro-based selection)

## Multidimensional arrays

TODO: compare the implementation of multidimensional arrays
(features described in the merge request: https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/merge_requests/18 )

- http://cpptruths.blogspot.cz/2011/10/multi-dimensional-arrays-in-c11.html
- http://www.nongnu.org/tensors/ (last commit in 2012)
- https://bitbucket.org/wlandry/ftensor/src
- [Eigen tensors](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md?at=default&fileviewer=file-view-default) - Many operations, expression templates, either pure-static or pure-dynamic sizes, only column-major format (row-major support is incomplete), little GPU support.
- [cudarrays](https://github.com/cudarrays/cudarrays) - Only up to 3D arrays, both static and dynamic, compile-time permutations using `std::tuple`.
- [RAJA](https://github.com/LLNL/RAJA) - No memory management, views are initialized with a raw pointer, index permutations are initialized at runtime, only dynamic dimensions.
- [Kokkos](https://github.com/kokkos/kokkos) - Configurable layout and default selection based on the memory/execution space, but only AoS and SoA are considered, even for `N > 2`. For parallel work there is only one leading dimension - it does not map to 2D or 3D CUDA grids.
- [CUV](https://github.com/deeplearningais/CUV) - Assumption that "everything is an n-dimensional array" (like Matlab), CPU and GPU support, column-major or row-major, integration with Python and Numpy.
