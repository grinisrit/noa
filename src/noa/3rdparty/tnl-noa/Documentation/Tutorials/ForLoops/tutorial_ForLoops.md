# For loops tutorial

[TOC]

## Introduction

This tutorial shows how to use different kind of for-loops implemented in TNL. Namely, they are:

* **Parallel for** is a for-loop which can be run in parallel, i.e. all iterations of the loop must be independent. Parallel for can be run on both multicore CPUs and GPUs.
* **n-dimensional parallel for** is an extension of common parallel for into higher dimensions.
* **Unrolled for** is a for-loop which is performed sequentially and it is explicitly unrolled by C++ templates. Iteration bounds must be static (known at compile time).
* **Static for** is a for-loop with static bounds (known at compile time) and indices usable in constant expressions.

## Parallel For

Basic _parallel for_ construction in TNL serves for hardware platform transparent expression of parallel for-loops.
The hardware platform is specified by a template parameter.
The loop is implemented as \ref TNL::Algorithms::ParallelFor and can be used as:

```
ParallelFor< Device >::exec( start, end, function, arguments... );
```

The `Device` can be either \ref TNL::Devices::Host or \ref TNL::Devices::Cuda.
The first two parameters define the loop bounds in the C style.
It means that there will be iterations for indices `start`, `start+1`, ..., `end-1`.
The `function` is a lambda function to be called in each iteration.
It is supposed to receive the iteration index and arguments passed to the _parallel for_ (the last arguments).

See the following example:

\include ParallelForExample.cpp

The result is:

\include ParallelForExample.out

## n-dimensional Parallel For

For-loops in higher dimensions can be performed similarly via \ref TNL::Algorithms::ParallelFor2D and \ref TNL::Algorithms::ParallelFor3D.
In the following example we build a 2D mesh function on top of \ref TNL::Containers::Vector.
Two dimensional indices `( i, j )` are mapped to the vector index `idx` as `idx = j * xSize + i`, where the mesh function has dimensions `xSize * ySize`.
The following simple example performs initiation of the mesh function with a constant value `c = 1.0`:

\include ParallelForExample-2D.cpp

Notice the parameters of the lambda function `init`.
The first parameter `i` changes more often than `j` and therefore the index mapping has the form `j * xSize + i` to access the vector elements sequentially on CPU and to fulfill coalesced memory accesses on GPU.
The for-loop is executed by calling `ParallelFor2D` with proper device.
The first four parameters are `startX, startY, endX, endY` and on CPU this is equivalent to the following embedded for-loops:

```cpp
for( Index j = startY; j < endY; j++ )
   for( Index i = startX; i < endX; i++ )
      f( i, j, args... );
```

where `args...` stand for additional arguments passed to the for-loop.
After the parameters defining the loops bounds, lambda function (`init` in this case) is passed, followed by additional arguments that are forwarded to the lambda function after the iteration indices.
In the example above there are no additional arguments, since the lambda function `init` captures all variables it needs to work with.

For completeness, we show modification of the previous example into 3D:

\include ParallelForExample-3D.cpp

## Unrolled For

\ref TNL::Algorithms::unrolledFor is a for-loop that it is explicitly unrolled via C++ templates when the loop is short (up to eight iterations).
The bounds of `unrolledFor` loops must be constant (i.e. known at the compile time).
It is often used with static arrays and vectors.

See the following example:

\include unrolledForExample.cpp

Notice that the unrolled for-loop works with a lambda function similar to parallel for-loop.
The bounds of the loop are passed as template parameters in the statement `Algorithms::unrolledFor< int, 0, Size >`.
The parameter of the `unrolledFor` function is the functor to be called in each iteration.
The function gets the loop index `i` only, see the following example:

The result looks as:

\include unrolledForExample.out

The effect of `unrolledFor` is really the same as usual for-loop.
The following code does the same as the previous example:

```cpp
for( int i = 0; i < Size; i++ )
{
   a[ i ] = b[ i ] + 3.14;
   sum += a[ i ];
};
```

The benefit of `unrolledFor` is mainly in the explicit unrolling of short loops which can improve performance in some situations.
The maximum length of loops that will be fully unrolled can be specified using the fourth template parameter as follows:

```cpp
Algorithms::unrolledFor< int, 0, Size, 16 >( ... );
```

`unrolledFor` can be used also in CUDA kernels.

## Static For

\ref TNL::Algorithms::staticFor is a generic for-loop whose iteration indices are usable in constant expressions (e.g. template arguments). It can be used as

```cpp
staticFor< int, 0, N >( f );
```

which results in the following sequence of function calls:

```cpp
f( std::integral_constant< 0 >{} );
f( std::integral_constant< 1 >{} );
f( std::integral_constant< 2 >{} );
f( std::integral_constant< 3 >{} );
...
f( std::integral_constant< N-1 >{} );
```

Notice that each iteration index is represented by its own distinct type using \ref std::integral_constant. Hence, the functor `f` must be generic, e.g. a _generic lambda expression_ such as in the following example:

\include staticForExample.cpp

The output looks as follows:

\include staticForExample.out
