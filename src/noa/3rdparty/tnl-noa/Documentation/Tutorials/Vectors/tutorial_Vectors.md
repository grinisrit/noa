# Vectors tutorial

[TOC]

## Introduction

This tutorial introduces vectors in TNL. `Vector`, in addition to `Array`, offers also basic operations from linear algebra. The reader will mainly learn how to do Blas level 1 operations in TNL. Thanks to the design of TNL, it is easier to implement, hardware architecture transparent and in some cases even faster then [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) or [cuBlas](https://developer.nvidia.com/cublas) implementation.

## Vectors

`Vector` is, similar to `Array`, templated class defined in namespace `TNL::Containers` having three template parameters:

* `Real` is type of data to be stored in the vector
* `Device` is the device where the vector is allocated. Currently it can be either `Devices::Host` for CPU or `Devices::Cuda` for GPU supporting CUDA.
* `Index` is the type to be used for indexing the vector elements.

`Vector`, unlike `Array`, requires that the `Real` type is numeric or a type for which basic algebraic operations are defined. What kind of algebraic operations is required depends on what vector operations the user will call. `Vector` is derived from `Array` so it inherits all its methods. In the same way the `Array` has its counterpart `ArraView`, `Vector` has `VectorView` which is derived from `ArrayView`. We refer to to [Arrays tutorial](../Arrays/tutorial_Arrays.md) for more details.

### Horizontal operations

By *horizontal* operations we mean vector expressions where we have one or more vectors as an input and a vector as an output. In TNL, this kind of operations is performed by the [Expression Templates](https://en.wikipedia.org/wiki/Expression_templates). It makes algebraic operations with vectors easy to do and very efficient at the same time. In some cases, one get even more efficient code compared to [Blas](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) and [cuBlas](https://developer.nvidia.com/cublas). See the following example.

\includelineno Expressions.cpp

Output is:

\include Expressions.out

The expression is evaluated on the same device where the vectors are allocated, this is done automatically. One cannot, however, mix vectors from different devices in one expression. Vector expression may contain any common function like the following:

| Expression                     | Meaning                                                     |
|--------------------------------|-------------------------------------------------------------|
| `v = TNL::min( expr1, expr2 )` |  `v[ i ] = min( expr1[ i ], expr2[ i ] )`                   |
| `v = TNL::max( expr1, expr2 )` |  `v[ i ] = max( expr1[ i ], expr2[ i ] )`                   |
| `v =  TNL::abs( expr )`        |  `v[ i ] = abs( expr[ i ] )`                                |
| `v =  TNL::sin( expr )`        |  `v[ i ] = sin( expr[ i ] )`                                |
| `v =  TNL::cos( expr )`        |  `v[ i ] = cos( expr[ i ] )`                                |
| `v =  TNL::tan( expr )`        |  `v[ i ] = tan( expr[ i ] )`                                |
| `v =  TNL::asin( expr )`       |  `v[ i ] = asin( expr[ i ] )`                               |
| `v =  TNL::acos( expr )`       |  `v[ i ] = acos( expr[ i ] )`                               |
| `v =  TNL::atan( expr )`       |  `v[ i ] = atan( expr[ i ] )`                               |
| `v =  TNL::sinh( expr )`       |  `v[ i ] = sinh( expr[ i ] )`                               |
| `v =  TNL::cosh( expr )`       |  `v[ i ] = cosh( expr[ i ] )`                               |
| `v =  TNL::tanh( expr )`       |  `v[ i ] = tanh( expr[ i ] )`                               |
| `v =  TNL::asinh( expr )`      |  `v[ i ] = asinh( expr[ i ] )`                              |
| `v =  TNL::acosh( expr )`      |  `v[ i ] = acosh( expr[ i ] )`                              |
| `v =  TNL::atanh( expr )`      |  `v[ i ] = atanh( expr[ i ] )`                              |
| `v =  TNL::exp( expr )`        |  `v[ i ] = exp( expr[ i ] )`                                |
| `v =  TNL::log( expr )`        |  `v[ i ] = log( expr[ i ] )`                                |
| `v =  TNL::log10( expr )`      |  `v[ i ] = log10( expr[ i ] )`                              |
| `v =  TNL::log2( expr )`       |  `v[ i ] = log2( expr[ i ] )`                               |
| `v =  TNL::sqrt( expr )`       |  `v[ i ] = sqrt( expr[ i ] )`                               |
| `v =  TNL::cbrt( expr )`       |  `v[ i ] = cbrt( expr[ i ] )`                               |
| `v =  TNL::pow( expr )`        |  `v[ i ] = pow( expr[ i ] )`                                |
| `v =  TNL::floor( expr )`      |  `v[ i ] = floor( expr[ i ] )`                              |
| `v =  TNL::ceil( expr )`       |  `v[ i ] = ceil( expr[ i ] )`                               |
| `v =  TNL::sign( expr )`       |  `v[ i ] = sign( expr[ i ] )`                               |

Where `v` is a result vector and `expr`, `expr1` and `expr2` are vector expressions. Vector expressions can be combined with vector views (\ref TNL::Containers::VectorView) as well.

### Vertical operations

By *vertical operations* we mean (parallel) reduction based operations where we have one vector expressions as an input and one value as an output. For example computing scalar product, vector norm or finding minimum or maximum of vector elements is based on reduction. See the following example.

\includelineno Reduction.cpp

Output is:

\include Reduction.out

The following table shows vertical operations that can be used on vector expressions:

| Expression                                   | Meaning                                                                                            |
|----------------------------------------------|----------------------------------------------------------------------------------------------------|
| `v =  TNL::min( expr )`                      | `v` is minimum of `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                       |
| `std::pair( v, i ) =  TNL::argMin( expr )`   | `v` is minimum of `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`, `i` is index of the smallest element. |
| `v =  TNL::max( expr )`                      | `v` is maximum of `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                       |
| `std::pair( v, i ) =  TNL::argMax( expr )`   | `v` is maximum of `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`, `i` is index of the largest element.  |
| `v =  TNL::sum( expr )`                      | `v` is sum of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                          |
| `v =  TNL::maxNorm( expr )`                  | `v` is maximal norm of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                 |
| `v =  TNL::l1Norm( expr )`                   | `v` is l1 norm of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                      |
| `v =  TNL::l2Norm( expr )`                   | `v` is l2 norm of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                      |
| `v =  TNL::lpNorm( expr, p )`                | `v` is lp norm of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                      |
| `v =  TNL::product( expr )`                  | `v` is product of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                      |
| `v =  TNL::logicalAnd( expr )`               | `v` is logical AND of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                  |
| `v =  TNL::logicalOr( expr )`                | `v` is logical OR of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                   |
| `v =  TNL::binaryAnd( expr )`                | `v` is binary AND of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                   |
| `v =  TNL::binaryOr( expr )`                 | `v` is binary OR of  `expr[ 0 ], expr[ 1 ] , .... expr[ n-1 ]`.                                    |

## Static vectors

Static vectors are derived from static arrays and so they are allocated on the stack and can be created in CUDA kernels as well. Their size is fixed as well and it is given by a template parameter. Static vector is a templated class defined in namespace `TNL::Containers` having two template parameters:

* `Size` is the array size.
* `Real` is type of numbers stored in the array.

The interface of StaticVectors is smillar to Vector. Probably the most important methods are those related with static vector expressions which are handled by expression templates. They make the use of static vectors simpel and efficient at the same time. See the following simple demonstration:

\include StaticVectorExample.cpp

The output looks as:

\include StaticVectorExample.out

## Distributed vectors

TODO
