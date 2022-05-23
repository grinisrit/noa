# Flexible (parallel) reduction and prefix-sum tutorial

[TOC]

## Introduction

This tutorial introduces flexible parallel reduction in TNL. It shows how to easily implement parallel reduction with user defined operations which may run on both CPU and GPU. Parallel reduction is a programming pattern appering very often in different kind of algorithms for example in scalar product, vector norms or mean value evaluation but also in sequences or strings comparison.

## Flexible parallel reduction

We will explain the *flexible parallel reduction* on several examples. We start with the simplest sum of sequence of numbers followed by more advanced problems like scalar product or vector norms.

### Sum

We start with simple problem of computing sum of sequence of numbers \f[ s = \sum_{i=1}^n a_i. \f] Sequentialy, such sum can be computed very easily as follows:

\includelineno SequentialSum.cpp

Doing the same in CUDA for GPU is, however, much more difficult (see. [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)). The final code has tens of lines and it is something you do not want to write again and again anytime you need to sum a series of numbers. Using TNL and C++ lambda functions we may do the same on few lines of code efficiently and independently on the hardware beneath. Let us first rewrite the previous example using the C++ lambda functions:

\includelineno SequentialSumWithLambdas.cpp

As can be seen, we split the reduction into two steps:

1. `fetch` reads the input data. Thanks to this lambda you can:
   1. Connect the reduction algorithm with given input arrays or vectors (or any other data structure).
   2. Perform operation you need to do with the input data.
   3. Perform another secondary operation simoultanously with the parallel reduction.
2. `reduction` is operation we want to do after the data fetch. Usually it is summation, multiplication, evaluation of minimum or maximum or some logical operation.

Putting everything together gives the following example:

\includelineno SumExample.cpp

Since TNL vectors cannot be pass to CUDA kernels and so they cannot be captured by CUDA lambdas, we must first get vector view from the vector using a method `getConstView()`.

Note tha we pass `0.0` as the last argument of the template function `reduce< Device >`. It is an *idempotent element* (see [Idempotence](https://cs.wikipedia.org/wiki/Idempotence)). It is an element which, for given operation, does not change the result. For addition, it is zero. The result looks as follows.

\include SumExample.out

Sum of vector elements can be also obtained as [`sum(v)`](../html/namespaceTNL.html#a41cea4796188f0877dbb6e72e2d3559e).

### Product

To demonstrate the effect of the *idempotent element*, we will now compute product of all elements of the vector. The *idempotent element* is one for multiplication and we also need to replace `a+b` with `a*b` in the definition of `reduction`. We get the following code:

\includelineno ProductExample.cpp

leading to output like this:

\include ProductExample.out

Product of vector elements can be computed using fuction [`product(v)`](../html/namespaceTNL.html#ac11e1901681d36b19a0ad3c6f167a718).

### Scalar product

One of the most important operation in the linear algebra is the scalar product of two vectors. Compared to coputing the sum of vector elements we must change the function `fetch` to read elements from both vectors and multiply them. See the following example.

\includelineno ScalarProductExample.cpp

The result is:

\include ScalarProductExample.out

Scalar product of vectors `u` and `v` in TNL can be computed by \ref TNL::dot "TNL::dot(u, v)" or simply as \ref TNL::Containers::operator, "(u, v)".

### Maximum norm

Maximum norm of a vector equals modulus of the vector largest element.  Therefore, `fetch` must return the absolute value of the vector elements and `reduction` wil return maximum of given values. Look at the following example.

\includelineno MaximumNormExample.cpp

The output is:

\include MaximumNormExample.out

Maximum norm in TNL is computed by the function \ref TNL::maxNorm.

### Vectors comparison

Comparison of two vectors involve (parallel) reduction as well. The `fetch` part is responsible for comparison of corresponding vector elements result of which is boolean `true` or `false` for each vector elements. The `reduction` part must perform logical and operation on all of them. We must not forget to change the *idempotent element* to `true`. The code may look as follows:

\includelineno ComparisonExample.cpp

And the output looks as:

\include ComparisonExample.out

### Update and residue

In iterative solvers we often need to update a vector and compute the update norm at the same time. For example the [Euler method](https://en.wikipedia.org/wiki/Euler_method) is defined as

\f[
\bf u^{k+1} = \bf u^k + \tau \Delta \bf u.
\f]

Together with the vector addition, we may want to compute also \f$L_2\f$-norm of \f$\Delta \bf u\f$ which may indicate convergence. Computing first the addition and then the norm would be inefficient because we would have to fetch the vector \f$\Delta \bf u\f$ twice from the memory. The following example shows how to do the addition and norm computation at the same time.

\includelineno UpdateAndResidueExample.cpp

The result reads as:

\include UpdateAndResidueExample.out

### Simple MapReduce

We can also filter the data to be reduced. This operation is called [MapReduce](https://en.wikipedia.org/wiki/MapReduce) . You simply add necessary if statement to the fetch function, or in the case of the following example we use a statement

```
return u_view[ i ] > 0.0 ? u_view[ i ] : 0.0;
```

to sum up only the positive numbers in the vector.

\includelineno MapReduceExample-1.cpp

The result is:

\include MapReduceExample-1.out

Take a look at the following example where the filtering depends on the element indexes rather than values:

\includelineno MapReduceExample-2.cpp

The result is:

\include MapReduceExample-2.out

This is not very efficient. For half of the elements, we return zero which has no effect during the reductin. Better solution is to run the reduction only for a half of the elements and to change the fetch function to

```
return u_view[ 2 * i ];
```

See the following example and compare the execution times.

\includelineno MapReduceExample-3.cpp

\include MapReduceExample-3.out

### Reduction with argument

In some situations we may need to locate given element in the vector. For example index of the smallest or the largest element. `reduceWithArgument` is a function which can do it. In the following example, we modify function for computing the maximum norm of a vector. Instead of just computing the value, now we want to get index of the element having the absolute value equal to the max norm. The lambda function `reduction` do not compute only maximum of two given elements anymore, but it must also compute index of the winner. See the following code:

\includelineno ReductionWithArgument.cpp

The definition of the lambda function `reduction` reads as:

```
auto reduction = [] __cuda_callable__ ( double& a, const double& b, int& aIdx, const int& bIdx );
```

In addition to vector elements values `a` and `b`, it gets also their positions `aIdx` and `bIdx`. The functions is responsible to set `a` to maximum of the two and `aIdx` to the position of the larger element. Note, that the parameters have the above mentioned meaning only in case of computing minimum or maximum.

The result looks as:

\include ReductionWithArgument.out

### Using functionals for reduction

You might notice, that the lambda function `reduction` does not take so many different form compared to fetch. In addition, setting the zero (or idempotent) element can be annoying especially when computing minimum or maximum and we need to check std::limits function to make the code working with any type. To make things simpler, TNL offers variants of several functionals known from STL. They can be used instead of the lambda function `reduction` and they also carry the idempotent element. See the following example showing the scalar product of two vectors, now with functional:

\includelineno ScalarProductWithFunctionalExample.cpp


This example also shows more compact how to evoke the function `reduce` (lines 19-22). This way, one should be able to perform (parallel) reduction very easily. The result looks as follows:

\include ScalarProductWithFunctionalExample.out

In \ref TNL/Functionals.h you may find probably all operations that can be reasonably used for reduction:

| Functional                      | Reduction operation      |
|---------------------------------|--------------------------|
| \ref TNL::Plus                  | Sum                      |
| \ref TNL::Multiplies            | Product                  |
| \ref TNL::Min                   | Minimum                  |
| \ref TNL::Max                   | Maximum                  |
| \ref TNL::MinWithArg            | Minimum with argument    |
| \ref TNL::MaxWithArg            | Maximum with argument    |
| \ref TNL::LogicalAnd            | Logical AND              |
| \ref TNL::LogicalOr             | Logical OR               |
| \ref TNL::BitAnd                | Bit AND                  |
| \ref TNL::BitOr                 | Bit OR                   |

## Flexible scan

### Inclusive and exclusive scan

Inclusive scan (or prefix sum) operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence \f$s_1, \ldots, s_n\f$ defined as

\f[
s_i = \sum_{j=1}^i a_i.
\f]

Exclusive scan (or prefix sum) is defined as

\f[
\sigma_i = \sum_{j=1}^{i-1} a_i.
\f]

For example, inclusive prefix sum of

```
[1,3,5,7,9,11,13]
```

is

```
[1,4,9,16,25,36,49]
```

and exclusive prefix sum of the same sequence is

```
[0,1,4,9,16,25,36]
```

Both kinds of [scan](https://en.wikipedia.org/wiki/Prefix_sum) have many different [applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) but they are usually applied only on summation, however product or logical operations could be handy as well. In TNL, prefix sum is implemented in similar way as reduction and so it can be easily modified by lambda functions. The following example shows how it works:

```
inplaceInclusiveScan( array, 0, array.getSize(), TNL::Plus{} );
```

This is equivalent to the following shortened call (the second, third and fourth parameters have a default value):

```
inplaceInclusiveScan( array );
```

The complete example looks as follows:

\includelineno inplaceInclusiveScanExample.cpp

Scan does not use `fetch` function because the scan must be performed on an array. Its complexity is also higher compared to reduction. Thus if one needs to do some operation with the array elements before the scan, this can be done explicitly and it will not affect the performance significantly. On the other hand, the scan function takes interval of the vector elements where the scan is performed as its second and third argument. The next argument is the operation to be performed by the scan and the last parameter is the idempotent ("zero") element of the operation.

The result looks as:

\include inplaceInclusiveScanExample.out

Exclusive scan works similarly. The complete example looks as follows:

\includelineno inplaceExclusiveScanExample.cpp

And the result looks as:

\include inplaceExclusiveScanExample.out

### Segmented scan

Segmented scan is a modification of common scan. In this case the sequence of numbers in hand is divided into segments like this, for example

```
[1,3,5][2,4,6,9][3,5],[3,6,9,12,15]
```

and we want to compute inclusive or exclusive scan of each segment. For inclusive segmented prefix sum we get

```
[1,4,9][2,6,12,21][3,8][3,9,18,30,45]
```

and for exclusive segmented prefix sum it is

```
[0,1,4][0,2,6,12][0,3][0,3,9,18,30]
```

In addition to common scan, we need to encode the segments of the input sequence. It is done by auxiliary flags array (it can be array of booleans) having `1` at the begining of each segment and `0` on all other positions. In our example, it would be like this:

```
[1,0,0,1,0,0,0,1,0,1,0,0, 0, 0]
[1,3,5,2,4,6,9,3,5,3,6,9,12,15]
```
**Note: Segmented scan is not implemented for CUDA yet.**

\includelineno SegmentedScanExample.cpp

The result reads as:

\include SegmentedScanExample.out
