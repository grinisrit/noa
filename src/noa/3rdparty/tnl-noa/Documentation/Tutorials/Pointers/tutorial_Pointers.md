# Cross-device smart pointers tutorial

[TOC]

## Introduction

Smart pointers in TNL are motivated by the smart pointers in the STL library. In addition, they can manage image of the object they hold on different devices which is supposed to make objects offloading easier.

## Unique pointers

Simillar to STL unique smart pointer `std::unique_ptr`, `UniquePointer` manages certain dynamicaly allocated object. The object is automatically deallocated when the pointer goes out of scope. The definition of `UniquePointer` reads as:

\include codeSnippetUniquePointer.cpp

It takes two template parameters:

1. `Object` is a type of object managed by the pointer.
2. `Device` is a device where the object is to be allocated.

If the device type is `Devices::Host`, `UniquePointer` behaves as usual unique smart pointer. See the following example:

\include UniquePointerHostExample.cpp

The result is:

\include UniquePointerHostExample.out


If the device is different, `Devices::Cuda` for example, the unique pointer creates an image of the object even in the host memory. It allows one to manipulate the object on the host. All smart pointers are registered in a special register using which they can be synchronised with the host images before calling a CUDA kernel - all at once. This means that all modified images of the objects in the host memory are transferred on the GPU. See the following example:

\include UniquePointerExample.cpp

The result looks as:

\include UniquePointerExample.out

A disadventage of `UniquePointer` is that it cannot be passed to the CUDA kernel since it requires making a copy of itself. This is, however, from the nature of this object, prohibited. For this reason we have to derreference the pointer on the host. This is done by a method `getData`. Its template parameter tells what object image we want to dereference - the one on the host or the one on the device. When we passing the object on the device, we need to get the device image. The method `getData` returns constant reference on the object. Non-constant reference is accessible via a method `modifyData`. When this method is used to get the reference on the host image, the pointer is marked as **potentialy modified**. Note that we need to have non-const reference even when we need to change the data (array elements for example) but not the meta-data (array size for example). If meta-data do not change there is no need to synchronize the object image with the one on the device. To distinguish between these two situations, the smart pointer keeps one more object image which stores the meta-data state since the last synchronization. Before the device image is synchronised, the host image and the last-synchronization-state image are compared. If they do not change no synchronization is required. One can see that TNL cross-device smart pointers are really meant only for small objects, otherwise the smart pointers overhead might be significant.

## Shared pointers

One of the main goals of the TNL library is to make the development of the HPC code, including GPU kernels, as easy and efficient as possible. One way to do this is to profit from the object opriented programming even in CUDA kernels. Let us explain it on arrays. From certain point of view `Array` can be understood as an object consisting of data and metadata. Data part means elements that we insert into the array. Metadata is a pointer to the data but also size of the array. This information makes use of the class easier for example by checking array bounds when accessing the array elements. It is something that, when it is performed even in CUDA kernels, may help significantly with finding bugs in a code. To do this, we need to transfer not only pointers to the data but also complete metadata on the device. It is simple if the structure which is supposed to be transfered on the GPU does not have pointers to metadata. See the following example:


\include codeSnippetSharedPointer-1.cpp

If the pointer `data` points to a memory on GPU, this array can be passed to a kernel like this:

\include codeSnippetSharedPointer-2.cpp

The kernel `cudaKernel` can access the data as follows:

\include codeSnippetSharedPointer-3.cpp

But what if we have an object like this:

\include codeSnippetSharedPointer-4.cpp

Assume that there is an instance of `ArrayTuple` lets say `tuple` containing pointers to instances `a1` and `a2` of `Array`. The instances must be allocated on the GPU if one wants to simply pass the `tuple` to the CUDA kernel. Indeed, the CUDA kernels needs the arrays `a1` and `a2` to be on the GPU. See the following example:

\include codeSnippetSharedPointer-5.cpp

See, that the kernel needs to dereference `tuple.a1` and `tuple.a2`. Therefore these pointers must point to the global memoty of the GPU which means that arrays `a1` and `a2` must be allocated there using [cudaMalloc](http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/group__CUDART__MEMORY_gc63ffd93e344b939d6399199d8b12fef.html) lets say. It means, however, that the arrays `a1` and `a2` cannot be managed (for example resizing them requires changing `a1->size` and `a2->size`) on the host system by the CPU. The only solution to this is to have images of `a1` and `a2` and in the host memory and to copy them on the GPU before calling the CUDA kernel. One must not forget to modify the pointers in the `tuple` to point to the array copies on the GPU. To simplify this, TNL offers *cross-device shared smart pointers*. In addition to common smart pointers thay can manage an images of an object on different devices. Note that [CUDA Unified Memory](https://devblogs.nvidia.com/unified-memory-cuda-beginners/) is an answer to this problem as well. TNL cross-device smart pointers can be more efficient in some situations. (TODO: Prove this with benchmark problem.)

The previous example could be implemented in TNL as follows:

\include SharedPointerExample.cpp

The result looks as:

\include SharedPointerExample.out

One of the differences between `UniquePointer` and `SmartPointer` is that the `SmartPointer` can be passed to the CUDA kernel. Dereferencing by operators `*` and `->` can be done in kernels as well and the result is reference to a proper object image i.e. on the host or the device. When these operators are used on constant smart pointer, constant reference is returned which is the same as calling the method `getData` with appropriate explicitely stated `Device` template parameter. In case of non-constant `SharedPointer` non-constant reference is obtained. It has the same effect as calling `modifyData` method. On the host system, everything what was mentioned in the section about `UniquePointer` holds even for the `SharedPointer`. In addition, `modifyData` method call or non-constant dereferencing can be done in kernel on the device. In this case, the programmer gets non-constant reference to an object which is however meant to be used to change the data managed by the object but not the metadata. There is no way to synchronize objects managed by the smart pointers from the device to the host. **It means that the metadata should not be changed on the device!** In fact, it would not make sense. Imagine changing array size or re-allocating the array within a CUDA kernel. This is something one should never do.

## Device pointers

The last type of the smart pointer implemented in TNL is `DevicePointer`. It works the same way as `SharedPointer` but it does not create new object on the host system. `DevicePointer` is therefore useful in situation when there is already an object created in the host memory and we want to create its image even on the device. Both images are linked one with each other and so one can just manipulate the one on the host and then synchronize it on the device. The following listing is a modification of the previous example with tuple:

\include DevicePointerExample.cpp

The result looks the same:

\include DevicePointerExample.out
