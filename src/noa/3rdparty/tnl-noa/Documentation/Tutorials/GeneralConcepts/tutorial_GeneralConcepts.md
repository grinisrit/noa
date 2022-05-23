# General concepts

[TOC]

## Introduction

In this part we describe some general and core concepts of programming with TNL. Understanding these ideas may significantly help to understand the design of TNL algorithms and data structure and it also helps to use TNL more efficiently. The main goal of TNL is to allow developing high performance algorithms that could run on multicore CPUs and GPUs. TNL offers unified interface and so the developer writes one code for both architectures.

## Devices and allocators

TNL offers unified interface for both CPUs (also referred as a host system) and GPUs (referred as device). Connection between CPU and GPU is usually represented by [PCI-Express bus](https://en.wikipedia.org/wiki/PCI_Express) which is orders of magnitude slower compared to speed of the global memory of GPU. Therefore, the communication between CPU and GPU must be reduced as much as possible. As a result, the programmer operates with two different address spaces, one for CPU and one for GPU. To distinguish between the address spaces, each data structure requiring dynamic allocation of memory needs to now on what device it resides. This is done by a template parameter `Device`. For example the following code creates two arrays, one on CPU and the other on GPU

\includelineno snippet_devices_and_allocators_arrays_example.cpp

Since now, [C++ template sepcialization](https://en.wikipedia.org/wiki/Partial_template_specialization) takes care of using the right methods for given device (in meaning hardware architecture and so the  device can be even CPU). For example, calling a method `setSize`

\includelineno snippet_devices_and_allocators_arrays_setsize_example.cpp

results in different memory allocation on CPU (for `host_array`) and on GPU (for `cuda_array`). The same holds for assignment

\includelineno snippet_devices_and_allocators_arrays_assignment_example.cpp

in which case appropriate data transfer from CPU to GPU is performed. Each such data structure contains inner type named `DeviceType` which tells where it resides as we can see here:

\includelineno snippet_devices_and_allocators_arrays_device_deduction.cpp

If we need to specialize some parts of algorithm with respect to its device we can do it by means of  \ref std::is_same :

\includelineno snippet_devices_and_allocators_arrays_device_check.cpp

TODO: Allocators

## Algorithms and lambda functions

Developing a code for GPUs (in [CUDA](https://developer.nvidia.com/CUDA-zone) for example) consists mainly of writing [kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) which are special functions running on GPU in parallel. This can be very hard and tedious work especially when it comes to debugging. [Parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) is a perfect example of an algorithm which is relatively hard to understand and implement on one hand but it is necessary to use frequently. Writing tens of lines of code every time we need to sum up some data is exactly what we mean by tedious programming. TNL offers skeletons or patterns of such algorithms and combines them with user defined [lambda functions](https://en.cppreference.com/w/cpp/language/lambda). This approach is not absolutely general, which means that you can use it only in situation when there is a skeleton/pattern (see \ref TNL::Algorithms) suitable for your problem. But when there is, it offers several advantages:

1. Implementing lambda functions is much easier compared to implementing GPU kernels.
2. Code implemented this way works even on CPU, so the developer writes only one code for both hardware architectures.
3. The developer may debug the code on CPU first and then just run it on GPU. Quite likely it will work with only a little or no changes.

The following code snippet demonstrates it on use of \ref TNL::Algorithms::ParallelFor:

\includelineno snippet_algorithms_and_lambda_functions_parallel_for.cpp

In this example, we assume that all arrays `v1`, `v2` and `sum` were properly allocated on given `Device`. If `Device` equals \ref TNL::Devices::Host , the lambda function is processed sequentially or in parallel by several OpenMP threads on CPU. If `Device` equals \ref TNL::Devices::Cuda , the lambda function is called from CUDA kernel (this is why it is defined as `__cuda_callable__` which is just a substitute for `__host__ __device__` ) by apropriate number of CUDA threads. One more example demonstrates use of \ref TNL::Algorithms::Reduction .

\includelineno snippet_algorithms_and_lambda_functions_reduction.cpp

We will not explain the parallel reduction in TNL at this moment (see the section about [flexible parallel reduction](../ReductionAndScan/tutorial_ReductionAndScan.md) ), we hope that the idea is more or less clear from the code snippet. If `Device` equals to \ref TNL::Device::Host , the scalar product is evaluated sequentially or in parallel by several OpenMP threads on CPU, if `Device` equals \ref TNL::Algorithms::Cuda, the [parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) fine tuned with the lambda functions is performed. Fortunately, there is no performance drop. On the contrary, since it is easy to generate CUDA kernels for particular situations, we may get more efficient code. Consider computing a scalar product of sum of vectors like this

\f[
s = (u_1 + u_2, v_1 + v_2).
\f]

This can be solved by the following code

\includelineno snippet_algorithms_and_lambda_functions_reduction_2.cpp

We have changed only the `fetch` lambda function to perform the sums of `u1[ i ] + u2[ i ]` and `v1[ i ] + v2[ i ]` (line 7). Now we get completely new CUDA kernel tailored exactly for our problem. Doing the same with [Cublas](https://developer.nvidia.com/cublas), for example, would require splitting into three separate kernels:

1. Kernel to compute \f$u_1 = u_1 + u_2\f$.
2. Kernel to compute \f$v_1 = v_1 + v_2\f$.
3. Kernel to compute \f$product = ( u_1, v_1 )\f$.

This could be achieved with the following code:

\includelineno snippet_algorithms_and_lambda_functions_reduction_cublas.cpp

We believe that C++ lambda functions with properly designed patterns of parallel algorithms could make programming of GPUs significantly easier. We see a parallel with [MPI standard](https://en.wikipedia.org/wiki/Message_Passing_Interface) which in nineties defined frequent communication operations in distributed parallel computing. It made programming of distributed systems much easier and at the same time MPI helps to write efficient programs. We aim to add additional skeletons or patterns to \ref TNL::Algorithms.

## Shared pointers and views

You might notice that in the previous section we used only C style arrays represented by pointers in the lambda functions. There is a difficulty when we want to access TNL arrays or other data structures inside the lambda functions. We may capture the outside variables either by a value or a reference. The first case would be as follows:

\includelineno snippet_shared_pointers_and_views_capture_value.cpp

In this case a deep copy of array `a` will be made and so there will be no effect of what we do with the array `a` in the lambda function. Capturing by a reference may look as follows:

\includelineno snippet_shared_pointers_and_views_capture_reference.cpp

This would be correct on CPU (i.e. when `Device` is \ref TNL::Devices::Host ). However, we are not allowed to pass references to CUDA kernels and so this source code would not even compile with CUDA compiler. To overcome this issue, TNL offers two solutions:

1. Data structures views
2. Shared pointers

### Data structures views

View is a kind of lightweight reference object which makes only a shallow copy of itself in copy constructor. Therefore view can by captured by value, but because it is, in fact, a reference to another object, everything we do with the view will affect the original object. The example with the array would look as follows:

\includelineno snippet_shared_pointers_and_views_capture_view.cpp

The differences are on the line 5 where we fetch the view by means of method `getView` and on the line 7 where we work with the `view` and not with the array `a`. The view has very similar interface (see \ref TNL::Containers::ArrayView) as the array (\ref TNL::Containers::Array) and so mostly there is no difference in using array and its view for the programmer. In TNL, each data structure which can be accessed from GPU kernels (it means that it has methods defined as `__cuda_callable__`) provides also a method `getView` for getting appropriate view of the object.

Views are simple objects because they must be transferred to GPU in each kernel call. So there are no smart links between a view and the original object. In fact, the array view contains just a pointer the data managed by the array and the size of the array. Therefore if the original object get changed, all views obtained from the object before may become invalid. See the following example:

\includelineno snippet_shared_pointers_and_views_capture_view_change.cpp

Such code would not work because after obtaining the view on the line 5, we change the size of the array `a` which will cause data reallocation. As we mentioned, there is no pointer from the view to the array and so the view has no chance to check if it is still up-to-date with the original object. However, if you fetch all necessary views immediately before capturing by a lambda function, there will be no problem. And this is why **the views are recommended for accessing TNL data structures in lambda functions or GPU kernels**.

Note, that changing the data managed by the array after fetching the view is not an issue. See the following example:

\includelineno snippet_shared_pointers_and_views_capture_view_change_2.cpp

On the line 6, we change value of the first element. This causes no data reallocation or change of size and so the view fetched on the line 5 is still valid and up-to-date.

### Shared pointers

TNL offers smart pointers working across different devices (meaning CPU or GPU).
