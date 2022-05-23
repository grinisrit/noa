# Arrays tutorial

[TOC]

## Introduction

This tutorial introduces arrays in TNL. There are three types - common arrays with dynamic allocation, static arrays allocated on stack and distributed arrays with dynamic allocation. Arrays are one of the most important structures for memory management. Methods implemented in arrays are particularly useful for GPU programming. From this point of view, the reader will learn how to easily allocate memory on GPU, transfer data between GPU and CPU but also, how to initialize data allocated on GPU. In addition, the resulting code is hardware platform independent, so it can be ran on CPU nad GPU without any changes.

## Arrays

Array is templated class defined in namespace `TNL::Containers` having three template parameters:

* `Value` is type of data to be stored in the array
* `Device` is the device where the array is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the array elements.

The following example shows how to allocate arrays on CPU and GPU and how to initialize the data.

\include ArrayAllocation.cpp

The result looks as follows:

\include ArrayAllocation.out


### Array views

Arrays cannot share data with each other or data allocated elsewhere. This can be achieved with the `ArrayView` structure which has similar semantics to `Array`, but it does not handle allocation and deallocation of the data. Hence, array view cannot be resized, but it can be used to wrap data allocated elsewhere (e.g. using an `Array` or an operator `new`) and to partition large arrays into subarrays. The process of wrapping external data with a view is called _binding_.

The following code snippet shows how to create an array view.

\include ArrayView-1.cpp

The output is:

\include ArrayView-1.out

You can also bind external data into array view:

\include ArrayView-2.cpp

Output:

\include ArrayView-2.out

Since array views do not allocate or deallocate memory, they can be created even in CUDA kernels, which is not possible with `Array`. `ArrayView` can also be passed-by-value into CUDA kernels or captured-by-value by device lambda functions, because the `ArrayView`'s copy-constructor makes only a shallow copy (i.e., it copies only the data pointer and size).

### Accessing the array elements

There are two ways how to work with the array (or array view) elements - using the indexing operator (`operator[]`) which is more efficient or using methods `setElement` and `getElement` which is more flexible.

#### Accessing the array elements with `operator[]`

Indexing operator `operator[]` is implemented in both `Array` and `ArrayView` and it is defined as `__cuda_callable__`. It means that it can be called even in CUDA kernels if the data is allocated on GPU, i.e. the `Device` parameter is `Devices::Cuda`. This operator returns a reference to given array element and so it is very efficient. However, calling this operator from host for data allocated on device (or vice versa) leads to segmentation fault (on the host system) or broken state of the device. It means:

* You may call the `operator[]` on the **host** only for data allocated on the **host** (with device `Devices::Host`).
* You may call the `operator[]` on the **device** only for data allocated on the **device** (with device `Devices::Cuda`).

The following example shows use of `operator[]`.

\include ElementsAccessing-1.cpp

Output:

\include ElementsAccessing-1.out

In general in TNL, each method defined as `__cuda_callable__` can be called from the CUDA kernels. The method `ArrayView::getSize` is another example. We also would like to point the reader to better ways of arrays initiation for example with method `ArrayView::forElements` or with `ParallelFor`.

#### Accessing the array elements with `setElement` and `getElement`

On the other hand, the methods `setElement` and `getElement` can be called from the host **no matter where the array is allocated**. In addition they can be called from kernels on device where the array is allocated. `getElement` returns copy of an element rather than a reference. Therefore it is slightly slower. If the array is on GPU and the methods are called from the host, the array element is copied from the device on the host (or vice versa) which is significantly slower. In the parts of code where the performance matters, these methods shall not be called from the host when the array is allocated on the device. In this way, their use is, however, easier compared to `operator[]` and they allow to write one simple code for both CPU and GPU. Both methods are good candidates for:

* reading/writing of only few elements in the array
* arrays initiation which is done only once and it is not time critical part of a code
* debugging purposes

The following example shows the use of `getElement` and `setElement`:

\include ElementsAccessing-2.cpp

Output:

\include ElementsAccessing-2.out

### Arrays and parallel for

More efficient and still quite simple method for (not only) array elements initiation is with the use of C++ lambda functions and methods `forElements` and `forAllElements`. As an argument a lambda function is passed which is then applied for all elements. Optionally one may define only subinterval of element indexes where the lambda shall be applied. If the underlying array is allocated on GPU, the lambda function is called from CUDA kernel. This is why it is more efficient than use of `setElement`. On the other hand, one must be careful to use only `__cuda_callable__` methods inside the lambda. The use of the methods `forElements` and `forAllElements` is demonstrated in the following example.

\include ArrayExample_forElements.cpp

Output:

\include ArrayExample_forElements.out

### Arrays and flexible reduction

Arrays also offer simpler way to do the flexible parallel reduction. See the section about [the flexible parallel reduction](../ReductionAndScan/tutorial_ReductionAndScan.md) to understand how it works. Flexible reduction for arrays just simplifies access to the array elements. See the following example:

\include ArrayExample_reduceElements.cpp

Output:

\include ArrayExample_reduceElements.out


### Checking the array contents

The functions \ref TNL::Algorithms::contains and \ref TNL::Algorithms::containsOnlyValue serve for testing the contents of arrays, vectors or their views.
`contains` returns `true` if there is at least one element in the array with given value.
`containsOnlyValue` returns `true` only if all elements of the array are equal to the given value.
The test can be restricted to a subinterval of array elements.
See the following code snippet for usage example.

\include contains.cpp

Output:

\include contains.out

### IO operations with arrays

Methods `save` and `load` serve for storing/restoring the array to/from a file in a binary form. In case of `Array`, loading of an array from a file causes data reallocation. `ArrayView` cannot do reallocation, therefore the data loaded from a file is copied to the memory managed by the `ArrayView`. The number of elements managed by the array view and those loaded from the file must be equal. See the following example.

\include ArrayIO.cpp

Output:

\include ArrayIO.out

## Static arrays

Static arrays are allocated on stack and thus they can be created even in CUDA kernels. Their size is fixed and it is given by a template parameter. Static array is a templated class defined in namespace `TNL::Containers` having two template parameters:

* `Size` is the array size.
* `Value` is type of data stored in the array.

The interface of StaticArray is very smillar to Array but much simpler. It contains set of common constructors. Array elements can be accessed by the `operator[]` and also using method `x()`, `y()` and `z()` when it makes sense. See the following example for typical use of StaticArray.

\include StaticArrayExample.cpp

The output looks as:

\include StaticArrayExample.out

## Distributed arrays

TODO
