# Matrices tutorial

[TOC]

TODO: Add description of forRows and sequentialForRows.

## Introduction

TNL offers several types of matrices like dense (\ref TNL::Matrices::DenseMatrix), sparse (\ref TNL::Matrices::SparseMatrix), tridiagonal (\ref TNL::Matrices::TridiagonalMatrix), multidiagonal (\ref TNL::Matrices::MultidiagonalMatrix) and lambda matrices (\ref TNL::Matrices::LambdaMatrix). The sparse matrices can be symmetric to lower the memory requirements. The interfaces of given matrix types are designed to be as unified as possible to ensure that the user can easily switch between different matrix types while making no or only a little changes in the source code. All matrix types allows traversing all matrix elements and manipulate them using lambda functions as well as performing flexible reduction in matrix rows. The following text describes particular matrix types and their unified interface in details.


## Overview of matrix types

In a lot of numerical algorithms either dense or sparse matrices are used. The dense matrix (\ref TNL::Matrices::DenseMatrix) is such that all or at least most of its matrix elements are nonzero. On the other hand [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) (\ref TNL::Matrices::SparseMatrix) is a matrix which has most of the matrix elements equal to zero. From the implementation point of view, the data structures for the dense matrices allocates all matrix elements while formats for the sparse matrices aim to store explicitly only the nonzero matrix elements. The most popular format for storing the sparse matrices is [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)). However, especially for better data alignment in memory of GPUs, many other formats were designed. In TNL, the user may choose between several different sparse matrix formats. There are also sparse matrices with specific pattern of the nonzero elements like [tridiagonal matrices](https://en.wikipedia.org/wiki/Tridiagonal_matrix) (\ref TNL::Matrices::TridiagonalMatrix) which "has nonzero elements on the main diagonal, the first diagonal below this, and the first diagonal above the main diagonal only". An example of such matrix may look as follows:

\f[
\left(
 \begin{array}{ccccccc}
 -2  &  1  &  .  & .   &  . & .  \\
  1  & -2  &  1  &  .  &  . & .  \\
  .  &  1  & -2  &  1   &  . & .  \\
  .  &  .  &  1  & -2  &  1 &  . \\
  .  &  .  &  .  &  1  & -2 &  1 \\
  .  &  .  &  .  &  .  &  1 & -2
 \end{array}
 \right)
\f]

Similar but more general type of matrices are multidiagonal matrices (\ref TNL::Matrices::MultidiagonalMatrix) which have the nonzero matrix elements positioned only on lines parallel to the main diagonal like the following matrix:

\f[
  \left(
  \begin{array}{ccccccc}
  -4  &  1  &  .  &  1  &  . & .  \\
   1  & -4  &  1  &  .  &  1 & .  \\
   .  &  1  & -4  &  1  &  . &  1 \\
   1  & .   &  1  & -4  &  1 &  . \\
   .  &  1  &  .  &  1  & -4 &  1 \\
   .  &  .  &  1  &  .  &  1 & -4
  \end{array}
  \right)
 \f]

Finally, TNL offers so called *lambda matrices* (\ref TNL::Matrices::LambdaMatrix) which are kind of "matrix-free matrices". They do not store the matrix elements explicitly in the memory, but rather evaluates them on-the-fly based on user defined lambda functions.

In the following table we show comparison of expressing a tridiagonal matrix by means of different matrix types.

| Matrix dimensions | Dense elems.   | Dense mem. | Sparse elems. | Sparse mem.  | Tridiag. elems. | Tridiag. mem. | Multidiag. elems. | Mutlidiag. mem. |
|------------------:|---------------:|-----------:|--------------:|-------------:|----------------:|--------------:|------------------:|----------------:|
|             10x10 |            100 |     800  B |           >28 |       >336 B |              30 |         240 B |                30 |           252 B |
|           100x100 |         10,000 |      80 kB |          >298 |     >3,576 B |             300 |       2,400 B |               300 |         2,412 B |
|       1,000x1,000 |      1,000,000 |       8 MB |        >2,998 |    >35,976 B |           3,000 |      24,000 B |             3,000 |        24,012 B |
|     10,000x10,000 |    100,000,000 |     800 MB |       >29,998 |   >359,976 B |          30,000 |     240,000 B |            30,000 |       240,012 B |
|   100,000x100,000 | 10,000,000,000 |      80 GB |      >299,998 | >3,599,876 B |         300,000 |   2,400,000 B |           300,000 |     2,400,012 B |

In the table:

* **Matrix dimensions** is the number of matrix rows and columns
* **Dense elems.** is the number of allocated matrix elements in the dense matrix.
* **Dense mem.** is the allocated memory for the matrix elements in the dense matrix if the elements are stored in the double precision.
* **Sparse elems.** is the number of allocated matrix elements in the sparse matrix. Some formats may allocate padding zeros for better data alignment in the memory and so the number of allocated matrix elements may increase.
* **Sparse mem.** is the allocated memory for the matrix elements in the sparse matrix if the elements are stored in the double precision and column indexes in 32-bit integer.
* **Tridiag. elems** is the number of allocated matrix elements in the tridiagonal matrix.
* **Tridiag mem.** is the allocated memory for the matrix elements in the tridiagonal matrix if the elements are stored in the double precision.
* **Multidiag. elems** is the number of allocated matrix elements in the multidiagonal matrix.
* **Multidiag mem.** is the allocated memory for the matrix elements in the multidiagonal matrix if the elements are stored in the double precision.

Choosing the best matrix type can have tremendous impact on the performance but also on memory requirements. If we would treat each matrix as dense one we would not be able to to work with matrices larger than 50,000x50,000 on common personal computers, because we would need tens of gibabytes of memory. At the same time, we see that the other matrix types can do the same job with only few megabytes. In addition, other matrix types work with much less matrix elements and so operations like matrix-vector multiplication can be done with significantly less operations which means much faster. Since in the modern hardware architectures, the computing units are limited mainly by the performance of the memory chips (so [called memory wall](https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall)), transferring less data from the memory increases the performance even more.

The following table shows the same as the one above but when storing a matrix which has only five nonzero elements in each row. Such matrices arise often from the finite difference method for solution of the partial differential equations:

| Matrix dimensions | Dense elems.   | Dense mem. | Sparse elems. | Sparse mem.  | Multidiag. elems. | Mutlidiag. mem. |
|------------------:|---------------:|-----------:|--------------:|-------------:|------------------:|----------------:|
|             10x10 |            100 |     800  B |           >50 |       >600 B |                50 |           420 B |
|           100x100 |         10,000 |      80 kB |          >500 |     >6,000 B |               500 |         4,020 B |
|       1,000x1,000 |      1,000,000 |       8 MB |        >5,000 |    >60,000 B |             5,000 |        40,020 B |
|     10,000x10,000 |    100,000,000 |     800 MB |       >50,000 |   >600,000 B |            50,000 |       400,020 B |
|   100,000x100,000 | 10,000,000,000 |      80 GB |      >500,000 | >6,000,000 B |           500,000 |     4,000,020 B |

There is no change in the dense matrix part of the table. The numbers grow proportionally in case of sparse and mutlidiagonal matrix. General sparse matrix formats need to store column indexes for each matrix element which is not true for the multidiagonal matrix. The following table shows how many bytes we need for storing of one matrix element with different matrix types depending on the type of the matrix elements (`Real`) and column indexes (`Index`):

| Real   | Index  | Dense matrix | Multidiagonal matrix |  Sparse matrix | Fill ratio |
|:------:|:------:|:------------:|:--------------------:|:--------------:|:----------:|
| float  | 32-bit |          4 B |                  4 B |            8 B |     << 50% |
| float  | 64-bit |          4 B |                  4 B |           12 B |     << 30% |
| double | 32-bit |          8 B |                  8 B |           12 B |     << 60% |
| double | 64-bit |          8 B |                  8 B |           16 B |     << 50% |

In this table:

* **Real** is matrix element type.
* **Index** is column index type.
* **Dense matrix** is number of bytes needed to store one matrix element in the dense matrix.
* **Multidiagonal matrix** is number of bytes needed to store one matrix element in the mutldiagonal matrix.
* **Sparse matrix** is number of bytes needed to store one matrix element in the sparse matrix.
* **Fill ratio** is maximal percentage of the nonzero matrix elements until which the sparse matrix can perform better.

The multidiagonal matrix type is especially suitable for the finite difference method or similar numerical methods for solution of the partial differential equations.

## Indexing of nonzero matrix elements in sparse matrices

The sparse matrix formats usually, in the first step, compress the matrix rows by omitting the zero matrix elements as follows

\f[
\left(
\begin{array}{ccccc}
0 & 1 & 0 & 2 & 0 \\
0 & 0 & 5 & 0 & 0 \\
4 & 0 & 0 & 0 & 7 \\
0 & 3 & 0 & 8 & 5 \\
0 & 5 & 7 & 0 & 0
\end{array}
\right)
\rightarrow
\left(
\begin{array}{ccccc}
1 & 2 & . & . & . \\
5 & . & . & . & . \\
4 & 7 & . & . & . \\
3 & 8 & 5 & . & . \\
5 & 7 & . & . & .
\end{array}
\right)
\f]

In such a form, it is more efficient to refer the nonzero matrix elements in given row by their rank in the compressed matrix row rather than by their column index in the original matrix. In methods for the sparse matrices, this parameter is called `localIdx`. Some sparse matrix formats add padding zeros for better alignment of data in memory. But if this is not the case, the variable `localIdx` of particular matrix elements would read as:

\f[
\left(
\begin{array}{ccccc}
0 & 1 & . & . & . \\
0 & . & . & . & . \\
0 & 1 & . & . & . \\
0 & 1 & 2 & . & . \\
0 & 1 & . & . & .
\end{array}
\right)
\f]

## Matrix view

Matrix views are small reference objects which help accessing the matrix in GPU kernels or lambda functions being executed on GPUs. We describe this in details in section about [Shared pointers and views](../GeneralConcepts/tutorial_GeneralConcepts.md). The problem lies in fact that we cannot pass references to GPU kernels and we do not want to pass there deep copies of matrices. Matrix view is some kind of reference to a matrix. A copy of matrix view is always shallow and so it behaves like a reference.  The following example shows how to obtain the matrix view by means of method `getView` and pass it to a lambda function:

\includelineno SparseMatrixViewExample_getRow.cpp

Here we create sparse matrix `matrix` on the line 11, and use the method `getView` to get the matrix view on the line 12. The view is then used in the lambda function on the line 15 where it is captured by value (see `[=]` in the definition of the lambda function `f` on the line 14).


## Allocation and setup of different matrix types

There are several ways how to create a new matrix:

1. **Initializer lists** allow to create matrix from the [C++ initializer lists](https://en.cppreference.com/w/cpp/utility/initializer_list). The matrix elements must be therefore encoded in the source code and so it is useful for rather smaller matrices. Methods and constructors with initializer lists are user friendly and simple to use. It is a good choice for tool problems with small matrices.
2. **STL map** can be used for creation of sparse matrices only. The user first insert all matrix elements together with their coordinates into [`std::map`](https://en.cppreference.com/w/cpp/container/map) based on which the sparse matrix is created in the next step. It is simple and user friendly approach suitable for creation of large matrices. An advantage is that we do not need to know the distribution of the nonzero matrix elements in matrix rows in advance like we do in other ways of construction of sparse matrices. This makes the use of STL map suitable for combining of sparse matrices from TNL with other numerical packages. However, the sparse matrix is constructed on the host and then copied on GPU if necessary. Therefore, this approach is not a good choice if fast and efficient matrix construction is required.
3. **Methods `setElement` and `addElement` called from the host** allow to change particular matrix elements. The methods can be called from host even for matrices allocated on GPU. In this case, however, the matrix elements are transferred on GPU one by one which is very inefficient. If the matrix is allocated on the host system (CPU), the efficiency is nearly optimal. In case of sparse matrices, one must set row capacities (i.e. maximal number of nonzero elements in each row) before using these methods. If the row capacity is exceeded, the matrix has to be reallocated and all matrix elements are lost.
4. **Methods `setElement` and `addElement` called on the host and copy matrix on GPU** setting particular matrix elements by the methods `setElement` and `addElement` when the matrix is allocated on GPU can be time consuming for large matrices. Setting up the matrix on CPU using the same methods and copying it on GPU at once when the setup is finished can be significantly more efficient. A drawback is that we need to allocate temporarily whole matrix on CPU.
5. **Methods `setElement` and `addElement` called from native device** allow to do efficient matrix elements setup even on devices (GPUs). In this case, the methods must be called from a GPU kernel or a lambda function combined with the parallel for (\ref TNL::Algorithms::ParallelFor). The user get very good performance even when manipulating matrix allocated on GPU. On the other hand, only data structures allocated on GPUs can be accessed from the kernel or lambda function. The matrix can be accessed in the GPU kernel or lambda function by means of [matrix view](#matrix_view) or the shared pointer (\ref TNL::Pointers::SharedPointer).
6. **Method `getRow` combined with `ParallelFor`** is very similar to the previous one. The difference is that we first fetch helper object called *matrix row* which is linked to particular matrix row. Using methods of this object, one may change the matrix elements in given matrix row. An advantage is that the access to the matrix row is resolved only once for all elements in the row. In some more sophisticated sparse matrix formats, this can be nontrivial operation and this approach may slightly improve the performance. Another advantage for sparse matrices is that we access the matrix elements based on their *local index* ('localIdx', see [Indexing of nonzero matrix elements in sparse matrices](indexing_of_nonzero_matrix_elements_in_sparse_matrices)) in the row which is something like a rank of the nonzero element in the row. This is more efficient than addressing the matrix elements by the column indexes which requires searching in the matrix row. So this may significantly improve the performance of setup of sparse matrices. When it comes to dense matrices, there should not be great difference in performance compared to use of the methods `setElement` and `getElement`. Note that when the method is called from a GPU kernel or a lambda function, only data structures allocated on GPU can be accessed and the matrix must be made accessible by the means of matrix view.
7. **Methods `forRows` and `forElements`** this approach is very similar to the previous one but it avoids using `ParallelFor` and necessity of passing the matrix to GPU kernels by matrix view or shared pointers.

The following table shows pros and cons of particular methods:

|  Method                                 | Efficient | Easy to use |  Pros                                                                 | Cons                                                                  |
|:----------------------------------------|:----------|:------------|:----------------------------------------------------------------------|:----------------------------------------------------------------------|
| **Initializer list**                    | **        | *****       | Very easy to use.                                                     | Only for small matrices.                                              |
|                                         |           |             | Does not need setting of matrix rows capacities                       |                                                                       |
| **STL map**                             | **        | *****       | Very easy to use.                                                     | Higher memory requirements.                                           |
|                                         |           |             | Does not need setting of matrix rows capacities                       | Slow transfer on GPU.                                                 |
| **[set,add]Element on host**            | ****/*    | *****       | Very easy to use.                                                     | Requires setting of row capacities.                                   |
|                                         |           |             |                                                                       | Extremely slow transfer on GPU.                                       |
| **[set,and]Element on host&copy on GPU**| ***       | ****        | Easy to use.                                                          | Requires setting of row capacities.                                   |
|                                         |           |             | Reasonable efficiency.                                                | Allocation of auxiliary matrix on CPU.                                |
| **[set,add]Element on native device**   | ****      |             | Good efficiency.                                                      | Requires setting of row capacities.                                   |
|                                         |           |             |                                                                       | Requires writing GPU kernel or lambda function.                       |
|                                         |           |             |                                                                       | Allows accessing only data allocated on the same device/memory space. |
| **getRow and ParallelFor**              | *****     | **          | Best efficiency for sparse matrices.                                  | Requires setting of row capacities.                                   |
|                                         |           |             |                                                                       | Requires writing GPU kernel or lambda function.                       |
|                                         |           |             |                                                                       | Allows accessing only data allocated on the same device/memory space. |
|                                         |           |             |                                                                       | Use of matrix local indexes can be less intuitive.                    |
| **forRows**, **forElements**            | *****     | **          | Best efficiency for sparse matrices.                                  | Requires setting of row capacities.                                   |
|                                         |           |             | Avoid use of matrix view or shared pointer in kernels/lambda function.| Requires writing GPU kernel or lambda function.                       |
|                                         |           |             |                                                                       | Allows accessing only data allocated on the same device/memory space. |
|                                         |           |             |                                                                       | Use of matrix local indexes is less intuitive.                        |

Though it may seem that the later methods come with more cons than pros, they offer much higher performance and we believe that even they are still user friendly. On the other hand, if the matrix setup performance is not a priority, the use easy-to-use but slow method can still be a good choice. The following tables demonstrate the performance of different methods. The tests were performed with the following setup:

|              |                                                   |
|--------------|---------------------------------------------------|
| CPU          | Intel i9-9900KF, 3.60GHz, 8 cores, 16384 KB cache |
| GPU          | GeForce RTX 2070                                  |
| g++ version  | 10.2.0                                            |
| nvcc version | 11.2.67                                           |
| Precision    | single precision                                  |

### Dense matrix

In the test of dense matrices, we set each matrix element to value equal to `rowIdx + columnIdx`. The times in seconds obtained on CPU looks as follows:

| Matrix rows and columns     | `setElement` on host | `setElement` with `ParallelFor` |  `getRow`    | `forElements`   |
|----------------------------:|---------------------:|--------------------------------:|-------------:|----------------:|
|                          16 |           0.00000086 |                       0.0000053 |   0.00000035 |       0.0000023 |
|                          32 |           0.00000278 |                       0.0000050 |   0.00000201 |       0.0000074 |
|                          64 |           0.00000703 |                       0.0000103 |   0.00000354 |       0.0000203 |
|                         128 |           0.00002885 |                       0.0000312 |   0.00000867 |       0.0000709 |
|                         256 |           0.00017543 |                       0.0000439 |   0.00002490 |       0.0001054 |
|                         512 |           0.00078153 |                       0.0001683 |   0.00005999 |       0.0002713 |
|                        1024 |           0.00271989 |                       0.0006691 |   0.00003808 |       0.0003942 |
|                        2048 |           0.01273520 |                       0.0038295 |   0.00039116 |       0.0017083 |
|                        4096 |           0.08381450 |                       0.0716542 |   0.00937997 |       0.0116771 |
|                        8192 |           0.51596800 |                       0.3535530 |   0.03971900 |       0.0467374 |

Here:

* **setElement on host** tests run in one thread. Therefore they are faster for small matrices compared to "`setElement` with `ParallelFor`" tests.
* **setElement with ParallelFor** tests run in parallel in several OpenMP threads. This approach is faster for larger matrices.
* **getRow** tests run in parallel in several OpenMP threads mapping of which is more efficient compared to "`setElement` on host" tests.

And the same on GPU is in the following table:

| Matrix rows and columns     | `setElement` on host | `setElement` on host and copy | `setElement` on GPU | `getRow`     | `forElements`   |
|----------------------------:|---------------------:|------------------------------:|--------------------:|-------------:|----------------:|
|                          16 |           0.027835   |                       0.02675 |         0.000101198 | 0.00009903   |     0.000101214 |
|                          32 |           0.002776   |                       0.00018 |         0.000099197 | 0.00009901   |     0.000100481 |
|                          64 |           0.010791   |                       0.00015 |         0.000094446 | 0.00009493   |     0.000101796 |
|                         128 |           0.043014   |                       0.00021 |         0.000099397 | 0.00010024   |     0.000102729 |
|                         256 |           0.171029   |                       0.00056 |         0.000100469 | 0.00010448   |     0.000105893 |
|                         512 |           0.683627   |                       0.00192 |         0.000103346 | 0.00011034   |     0.000112752 |
|                        1024 |           2.736680   |                       0.00687 |         0.000158805 | 0.00016932   |     0.000170302 |
|                        2048 |          10.930300   |                       0.02474 |         0.000509000 | 0.00050917   |     0.000511183 |
|                        4096 |          43.728700   |                       0.13174 |         0.001557030 | 0.00156117   |     0.001557930 |
|                        8192 |         174.923000   |                       0.70602 |         0.005312470 | 0.00526658   |     0.005263870 |

Here:

* **setElement on host** tests are very slow especially for large matrices since each matrix element is copied on GPU separately.
* **setElement on host and copy** tests are much faster because the matrix is copied from CPU to GPU on the whole which is more efficient.
* **setElement on GPU** tests are even more faster since there is no transfer of data between CPU and GPU.
* **getRow** tests have the same performance as "`setElement` on GPU".
* **forElements** tests have the same performance as both "`setElement` on GPU" and "`getRow`".

You can see the source code of the previous benchmark in [Appendix](#benchmark-of-dense-matrix-setup).

### Sparse matrix

The sparse matrices are tested on computation of matrix the [discrete Laplace operator in 2D](https://en.wikipedia.org/wiki/Discrete_Laplace_operator). This matrix has at most five nonzero elements in each row. The times for sparse matrix (with CSR format) on CPU in seconds looks as follows:

| Matrix rows and columns     |  STL Map     | `setElement` on host | `setElement` with `ParallelFor` | `getRow`    | `forElements`    |
|----------------------------:|-------------:|---------------------:|--------------------------------:|------------:|-----------------:|
|                         256 |      0.00016 |             0.000017 |                        0.000014 |    0.000013 |         0.000020 |
|                       1,024 |      0.00059 |             0.000044 |                        0.000021 |    0.000019 |         0.000022 |
|                       4,096 |      0.00291 |             0.000130 |                        0.000031 |    0.000022 |         0.000031 |
|                      16,384 |      0.01414 |             0.000471 |                        0.000067 |    0.000031 |         0.000065 |
|                      65,536 |      0.06705 |             0.001869 |                        0.000218 |    0.000074 |         0.000209 |
|                     262,144 |      0.31728 |             0.007436 |                        0.000856 |    0.000274 |         0.000799 |
|                   1,048,576 |      1.46388 |             0.027087 |                        0.006162 |    0.005653 |         0.005904 |
|                   4,194,304 |      7.46147 |             0.102808 |                        0.028385 |    0.027925 |         0.027937 |
|                  16,777,216 |     38.95900 |             0.413823 |                        0.125870 |    0.124588 |         0.123858 |
|                  67,108,864 |    185.75700 |             1.652580 |                        0.505232 |    0.501003 |         0.500927 |

Here:

* **STL Map** tests show that use of STL Map can be very slow on large matrices and, of course, they need to allocate the map containing all the matrix elements. This can be memory consuming. On the other hand, it is the only way which does not require knowing the matrix row capacities in advance.
* **setElement on host** tests are much faster compared to STL map, it does not need to allocate anything else except the sparse matrix. However, matrix row capacities must be known in advance.
* **setElement with ParallelFor** tests run in parallel in several OpenMP threads and so this can be faster for larger matrices.
* **getRow** tests perform the same as "setElement with ParallelFor".
* **forElements** tests perform the same as both "setElement with ParallelFor" and "forElements".

We see, that the use of STL map makes sense only in situation when it is hard to estimate necessary row capacities. Otherwise very easy setup with `setElement` method is much faster. If the performance is the highest priority, `getRow` method should be preferred. The results for GPU are in the following table:

| Matrix rows and columns     |  STL Map     | `setElement` on host | `setElement` on host and copy |`setElement` on GPU | `getRow`    | `forElements`   |
|----------------------------:|-------------:|---------------------:|------------------------------:|-------------------:|------------:|----------------:|
|                         256 |       0.002  |                0.036 |                        0.0280 |            0.00017 |     0.00017 |         0.00017 |
|                       1,024 |       0.001  |                0.161 |                        0.0006 |            0.00017 |     0.00017 |         0.00017 |
|                       4,096 |       0.003  |                0.680 |                        0.0010 |            0.00020 |     0.00020 |         0.00020 |
|                      16,384 |       0.015  |                2.800 |                        0.0034 |            0.00021 |     0.00020 |         0.00021 |
|                      65,536 |       0.074  |               11.356 |                        0.0130 |            0.00048 |     0.00047 |         0.00048 |
|                     262,144 |       0.350  |               45.745 |                        0.0518 |            0.00088 |     0.00087 |         0.00088 |
|                   1,048,576 |       1.630  |              183.632 |                        0.2057 |            0.00247 |     0.00244 |         0.00245 |
|                   4,194,304 |       8.036  |              735.848 |                        0.8119 |            0.00794 |     0.00783 |         0.00788 |
|                  16,777,216 |      41.057  |             2946.610 |                        3.2198 |            0.02481 |     0.02429 |         0.02211 |
|                  67,108,864 |     197.581  |            11791.601 |                       12.7775 |            0.07196 |     0.06329 |         0.06308 |

Here:

* **STL Map** tests show that the times are comparable to CPU times which means the most of the time is spent by creating the matrix on CPU.
* **setElement on host**  tests are again extremely slow for large matrices. It is even slower than the use of STL map. So in case of GPU, this is another reason for using the STL map.
* **setElement on host and copy** tests are, similar to the dense matrix, much faster compared to the previous approaches. So it is the best way when you need to use data structures available only on the host system (CPU).
* **setElement on GPU** tests exhibit the best performance together with `getRow` and `forElements` methods. Note, however, that this method can be slower that `getRow` and `forElements` if there would be more nonzero matrix elements in a row.
* **getRow** tests exhibit the best performance together with `setElement` on GPU and `forElements` methods.
* **forElements** tests exhibit the best performance together with `getRow` and `setElement` on GPU methods.

Here we see, that the `setElement` methods performs extremely bad because all matrix elements are transferred to GPU one-by-one. Even STL map is much faster. Note, that the times for STL map are not much higher compared to CPU which indicates that the transfer of the matrix on GPU is not dominant. Setup of the matrix on CPU by the means of `setElement` method and transfer on GPU is even faster. However, the best performance can be obtained only we creating the matrix directly on GPU by methods `setElement`, `getRow` and `forElements`. Note, however, that even if all of them perform the same way, for matrices with more nonzero matrix elements in a row, `setElement` could be slower compared to the `getRow` and `forElements`.

You can see the source code of the previous benchmark in [Appendix](#benchmark-of-sparse-matrix-setup).

### Multidiagonal matrix

Finally, the following tables show the times of the same test performed with multidiagonal matrix. Times on CPU in seconds looks as follows:

| Matrix rows and columns     |  `setElement` on host     | `setElement` with `ParallelFor` | `getRow`    | `forElements`   |
|----------------------------:|--------------------------:|--------------------------------:|------------:|----------------:|
|                         256 |                  0.000055 |                       0.0000038 |    0.000004 |        0.000009 |
|                       1,024 |                  0.000002 |                       0.0000056 |    0.000003 |        0.000006 |
|                       4,096 |                  0.000087 |                       0.0000130 |    0.000005 |        0.000014 |
|                      16,384 |                  0.000347 |                       0.0000419 |    0.000010 |        0.000046 |
|                      65,536 |                  0.001378 |                       0.0001528 |    0.000032 |        0.000177 |
|                     262,144 |                  0.005504 |                       0.0006025 |    0.000131 |        0.000711 |
|                   1,048,576 |                  0.019392 |                       0.0028773 |    0.001005 |        0.003265 |
|                   4,194,304 |                  0.072078 |                       0.0162378 |    0.011915 |        0.018065 |
|                  16,777,216 |                  0.280085 |                       0.0642682 |    0.048876 |        0.072084 |
|                  67,108,864 |                  1.105120 |                       0.2427610 |    0.181974 |        0.272579 |

Here:

* **setElement on host** tests show that this method is fairly efficient.
* **setElement with ParallelFor** tests run in parallel in several OpenMP threads compared to "setElement on host" tests. For larger matrices, this way of matrix setup performs better.
* **getRow** tests perform more or less the same as "setElement with ParallelFor" and `forElements`.
* **forElements** tests perform more or less the same as "setElement with ParallelFor" and `getRow`.

Note, that setup of multidiagonal matrix is faster compared to the same matrix stored in general sparse format. Results for GPU are in the following table:

| Matrix rows and columns     | `setElement` on host | `setElement` on host and copy | `setElement` on GPU | `getRow`    | `forElements`   |
|----------------------------:|---------------------:|------------------------------:|--------------------:|------------:|----------------:|
|                         256 |                0.035 |                       0.02468 |            0.000048 |    0.000045 |       0.000047  |
|                       1,024 |                0.059 |                       0.00015 |            0.000047 |    0.000045 |       0.000047  |
|                       4,096 |                0.251 |                       0.00044 |            0.000048 |    0.000045 |       0.000047  |
|                      16,384 |                1.030 |                       0.00158 |            0.000049 |    0.000046 |       0.000048  |
|                      65,536 |                4.169 |                       0.00619 |            0.000053 |    0.000048 |       0.000052  |
|                     262,144 |               16.807 |                       0.02187 |            0.000216 |    0.000214 |       0.000217  |
|                   1,048,576 |               67.385 |                       0.08043 |            0.000630 |    0.000629 |       0.000634  |
|                   4,194,304 |              270.025 |                       0.31272 |            0.001939 |    0.001941 |       0.001942  |
|                  16,777,216 |             1080.741 |                       1.18849 |            0.003212 |    0.004185 |       0.004207  |
|                  67,108,864 |             4326.120 |                       4.74481 |            0.013672 |    0.022494 |       0.030369  |

* **setElement on host** tests are extremely slow again, especially for large matrices.
* **setElement on host and copy** tests are much faster compared to the previous.
* **setElement with ParallelFor** tests offer the best performance. They are even faster then `getRow` and `forElements` method. This, however, does not have be true for matrices having more nonzero elements in a row.
* **getRow** tests perform more or less the same as `forElements`. For matrices having more nonzero elements in a row this method could be faster than `setElement`.
* **forElements** tests perform more or less the same as `getRow`.

Note that multidiagonal matrix performs better compared to general sparse matrix. One reason for it is the fact, that the multidiagonal type does not store explicitly column indexes of all matrix elements. Because of this, less data need to be transferred from the memory.

You can see the source code of the previous benchmark in [Appendix](#benchmark-of-multidiagonal-matrix-setup).

In the following parts we will describe hoe to setup particular matrix types by means of the methods mentioned above.

### Dense matrices

Dense matrix (\ref TNL::Matrices::DenseMatrix) is a templated class defined in the namespace \ref TNL::Matrices. It has five template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for CUDA supporting GPUs. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default, it is the row-major order if the matrix is allocated on the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.

The following examples show how to allocate the dense matrix and how to initialize the matrix elements.

#### Initializer list

Small matrices can be created simply by the constructor with an [initializer list](https://en.cppreference.com/w/cpp/utility/initializer_list).

\includelineno Matrices/DenseMatrix/DenseMatrixExample_Constructor_init_list.cpp

In fact, the constructor takes a list of initializer lists. Each embedded list defines one matrix row and so the number of matrix rows is given by the size of the outer initializer list.  The number of matrix columns is given by the longest inner initializer lists. Shorter inner lists are filled with zeros from the right side. The result looks as follows:

\include DenseMatrixExample_Constructor_init_list.out

#### Methods `setElement` and `addElement`

Larger matrices can be setup with methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement). The following example shows how to call these methods from the host.

\includelineno DenseMatrixExample_addElement.cpp

As we can see, both methods can be called from the host no matter where the matrix is allocated. If it is on GPU, each call of `setElement` or `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) causes slow transfer of tha data between CPU and GPU. Use this approach only if the performance is not a priority. The result looks as follows:

\include DenseMatrixExample_addElement.out

More efficient way of the matrix initialization on GPU consists of calling the methods `setElement` and `addElement` (\ref TNL::Matrices::DenseMatrix::setElement, \ref TNL::Matrices::DenseMatrix::addElement) directly from GPU, for example by means of lambda function and `ParallelFor2D` (\ref TNL::Algorithms::ParallelFor2D). It is demonstrated in the following example (of course it works even on CPU):

\includelineno DenseMatrixViewExample_setElement.cpp

Here we get the matrix view (\ref TNL::Matrices::DenseMatrixView) (line 10) to make the matrix accessible in lambda function even on GPU (see [Shared pointers and views](../GeneralConcepts/tutorial_GeneralConcepts.md) ). We first call the `setElement` method from CPU to set the `i`-th diagonal element to `i` (lines 11-12). Next we iterate over the matrix rows with `ParallelFor2D` (\ref TNL::Algorithms::ParallelFor2D) (line 20) and for each row we call the lambda function `f`. This is done on the same device where the matrix is allocated and so it we get optimal performance even for matrices on GPU. In the lambda function we add one to each matrix element (line 18). The result looks as follows:

\include DenseMatrixExample_setElement.out

#### Method `getRow`

This method is available for the dense matrix (\ref TNL::Matrices::DenseMatrix::getRow) mainly for two reasons:

1. The method `getRow` is recommended for sparse matrices. In most cases, it is not optimal for dense matrices. However, if one needs to have one code for both dense and sparse matrices, this method is a good choice.
2. In general, use of `setElement` (\ref TNL::Matrices::DenseMatrix::setElement) combined with `ParallelFor2D` (\ref TNL::Algorithms::ParallelFor2D) is preferred, for dense matrices, since it offers more parallelism for GPUs. `ParallelFor2D` creates one CUDA thread per each matrix element which is desirable for GPUs. With the use of the method `getRow` we have only one CUDA thread per each matrix row. This makes sense only in situation when we need to setup each matrix row sequentially.

Here we show an example:

\includelineno DenseMatrixViewExample_getRow.cpp

Here we create the matrix on the line 10 and get the matrix view on the line 16. Next we use `ParallelFor` (\ref TNL::Algorithms::ParallelFor) (line 31) to iterate over the matrix rows and call the lambda function `f` (lines 19-26) for each of them. In the lambda function, we first fetch the matrix row by means of the method `getRow` (\ref TNL::Matrices::DenseMatrixView::getRow) and next we set the matrix elements by using the method `setElement` of the matrix row (\ref TNL::Matrices::DenseMatrixRowView::setElement). For the compatibility with the sparse matrices, use the variant of `setElement` with the parameter `localIdx`. It has no effect here, it is only for compatibility of the interface.

The result looks as follows:

\include DenseMatrixViewExample_getRow.out

#### Method `forRows`

This method iterates in parallel over all matrix rows. In fact, it combines \ref TNL::Algorithms::ParallelFor and \ref TNL::Matrices:::DenseMatrix::getRow method in one. See the following example. It is even a bit simpler compared to the previous one:

\includelineno DenseMatrixExample_forRows.cpp

The lambda function `f`, which is called for each matrix row (lines 18-25), have to accept parameter `row` with type `RowView`. This type is defined inside each TNL matrix and in the case of the dense matrix, it is \ref TNL::Matrices::DenseMatrixRowView. We use the method \ref TNL::Matrices::DenseMatrixRowView::getRowIndex to get the index of the matrix row being currently processed and method \ref TNL::Matrices::DenseMatrixRowView::setElement which sets the value of the element with given column index (the first parameter).

Next, on the lines 32-38, we call another lambda function which firstly find the largest element in each row (lines 33-35) and then it divides the matrix row by its value (lines 36-37).

The result looks as follows:

\include DenseMatrixExample_forRows.out

#### Method `forElements`

 The next example demonstrates the method `forElements` (\ref TNL::Matrices::DenseMatrix::forElements) which works in very similar way as the method `getRow` but it is slightly easier to use. It is also compatible with sparse matrices. See the following example:

\includelineno DenseMatrixExample_forElements.cpp

We do not need any matrix view and instead of calling `ParallelFor` (\ref TNL::Algorithms::ParallelFor) we call just the method `forElements` (line 18). The lambda function `f` (line 11) must accept the following parameters:

* `rowIdx` is the row index of given matrix element.
* `columnIdx` is the column index of given matrix element.
* `value` is a reference on the matrix element value and so by changing this value we can modify the matrix element.
* `compute` is a boolean which, when set to `false`, indicates that we can skip the rest of the matrix row. This is, however, only a hint and it does not guarantee that the rest of the matrix row is really skipped.

The result looks as follows:

\include DenseMatrixExample_forElements.out

#### Wrapping existing data to dense matrix view

In case when you have already allocated data for dense matrix (for example in some other library), you may wrap it to dense matrix view with a function \ref TNL::Matrices::wrapDenseMatrix . See the following example:

\includelineno DenseMatrixViewExample_wrap.cpp

Here we create dense matrix having three rows and four columns. We use TNL vector (\ref TNL::Containers::Vector) only for allocation of the matrix elements (lines 12-15) and we get a pointer to the allocated array immediately (line 16). Next we use just the array to get dense matrix view with proper matrix dimensions (line 21). Note that we must explicitly state the device type as a template parameter of the function `wrapDenseMatrix` (\ref TNL::Matrices::wrapDenseMatrix). Finally, we print the matrix to see if it is correct (line 22). The result looks as follows:

\include DenseMatrixViewExample_wrap.out

### Sparse matrices

[Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) are extremely important in a lot of numerical algorithms. They are used at situations when we need to operate with matrices having majority of the matrix elements equal to zero. In this case, only the non-zero matrix elements are stored with possibly some *padding zeros* used for memory alignment. This is necessary mainly on GPUs. See the [Overview of matrix types](#overview_of_matrix_types) for the differences in memory requirements.

Major disadvantage of sparse matrices is that there are a lot of different formats for their storage in memory. Though [CSR (Compressed Sparse Row)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format is the most popular of all, especially for GPUs, there are many other formats. Often their performance differ significantly for various matrices. So it is a good idea to test several sparse matrix formats if you want to get the best performance. In TNL, there is one templated class \ref TNL::Matrices::SparseMatrix representing general sparse matrices. The change of underlying matrix format can be done just by changing one template parameter. The list of the template parameters is as follows:

* `Real` is type if the matrix elements. It is `double` by default.
* `Device` is a device where the matrix is allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for CUDA supporting GPUs. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `MatrixType` tells if the matrix is symmetric (\ref TNL::Matrices::SymmetricMatrix) or general (\ref TNL::Matrices::GeneralMatrix). It is a \ref TNL::Matrices::GeneralMatrix by default.
* `Segments` define the format of the sparse matrix. It can be one of the following (by default, it is \ref TNL::Algorithms::Segments::CSR):
   * \ref TNL::Algorithms::Segments::CSR for [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).
   * \ref TNL::Algorithms::Segments::Ellpack for [Ellpack format](http://mgarland.org/files/papers/nvr-2008-004.pdf).
   * \ref TNL::Algorithms::Segments::SlicedEllpack for [SlicedEllpack format](https://link.springer.com/chapter/10.1007/978-3-642-11515-8_10) which was also presented as [Row-grouped CSR format](https://arxiv.org/abs/1012.2270).
   * \ref TNL::Algorithms::Segments::ChunkedEllpack for [ChunkedEllpack format](http://geraldine.fjfi.cvut.cz/~oberhuber/data/vyzkum/publikace/12-heller-oberhuber-improved-rgcsr-format.pdf) which we reffered as Improved Row-grouped CSR and we renamed it to Ellpack format since it uses padding zeros.
   * \ref TNL::Algorithms::Segments::BiEllpack for [BiEllpack format](https://www.sciencedirect.com/science/article/pii/S0743731514000458?casa_token=2phrEj0Ef1gAAAAA:Lgf6rMBUN6T7TJne6mAgI_CSUJ-jR8jz7Eghdv6L0SJeGm4jfso-x6Wh8zgERk3Si7nFtTAJngg).
* `ComputeReal` is type which is used for internal computations. By default it is the same as `Real` if `Real` is not `bool`. If `Real` is `bool`, `ComputeReal` is set to `Index` type. This can be changed, of course, by the user.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type – see \ref TNL::Allocators::Default.
* `IndexAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the column indexes of the matrix elements. By default, it is the default allocator for given `Index` type and `Device` type – see \ref TNL::Allocators::Default.

**If `Real` is set to `bool`, we get *a binary matrix* for which the non-zero elements can be equal only to one and so the matrix elements values are not stored explicitly in the memory.** This can significantly reduce the memory requirements and also increase performance.

In the following text we will show how to create and setup sparse matrices.

#### Setting of row capacities

Larger sparse matrices are created in two steps:

1. We use a method \ref TNL::Matrices::SparseMatrix::setRowCapacities to initialize the underlying matrix format and to allocate memory for the matrix elements. This method only needs to know how many non-zero elements are supposed to be in each row. Once this is set, it cannot be changed only by resetting the whole matrix. In most situations, it is not an issue to compute the number of nonzero elements in each row. Otherwise, we can currently only recommend the use of matrix setup with [STL map](#sparse-matrix-stl-map), which is, however, quite slow.
2. Now, the nonzero matrix elements can be set one after another by telling its coordinates and a value. Since majority of sparse matrix formats are designed to allow quick access to particular matrix rows the insertion can be done in parallel by mapping different threads to different matrix rows. This approach is usually optimal or nearly optimal when it comes to efficiency.

See the following example which creates lower triangular matrix like this one

\f[
\left(
\begin{array}{ccccc}
 1 &  0 &  0 &  0 &  0 \\
 2 &  1 &  0 &  0 &  0 \\
 3 &  2 &  1 &  0 &  0 \\
 4 &  3 &  2 &  1 &  0 \\
 5 &  4 &  3 &  2 &  1
\end{array}
\right).
\f]

\includelineno SparseMatrixExample_setRowCapacities.cpp

The method \ref TNL::Matrices::SparseMatrix::setRowCapacities reads the required capacities of the matrix rows from a vector (or simmilar container - \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector and \ref TNL::Containers::VectorView) which has the same number of elements as the number of matrix rows and each element defines the capacity of the related row. The result looks as follows:

\include SparseMatrixExample_setRowCapacities.out

There are constructors which also set the row capacities. The first one uses a vector:

\includelineno SparseMatrixExample_Constructor_rowCapacities_vector.cpp

The second one uses an initializer list:

\includelineno SparseMatrixExample_Constructor_init_list_1.cpp

The result of both examples looks as follows:

\include SparseMatrixExample_Constructor_init_list_1.out

#### Initializer list

Small matrices can be initialized by a constructor with an [initializer list](https://en.cppreference.com/w/cpp/utility/initializer_list). We assume having the following sparse matrix

\f[
\left(
\begin{array}{ccccc}
 1 &  0 &  0 &  0 &  0 \\
-1 &  2 & -1 &  0 &  0 \\
 0 & -1 &  2 & -1 &  0 \\
 0 &  0 & -1 &  2 & -1 \\
 0 &  0 &  0 & -1 &  0
\end{array}
\right).
\f]

It can be created with the initializer list constructor like we shows in the following example:

\includelineno SparseMatrixExample_Constructor_init_list_2.cpp

The constructor accepts the following parameters (lines 9-17):

* `rows` is a number of matrix rows.
* `columns` is a number of matrix columns.
* `data` is definition of nonzero matrix elements. It is a initializer list of triples having a form `{ row_index, column_index, value }`. In fact, it is very much like the Coordinate format - [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)).

The constructor also accepts `Real` and `Index` allocators (\ref TNL::Allocators) but the default ones are used in this example. A method `setElements` (\ref TNL::Matrices::SparseMatrix::setElements) works the same way:

\includelineno SparseMatrixExample_setElements.cpp

In this example, we create the matrix in two steps. Firstly we use constructor with only matrix dimensions as parameters (line 9) and next we set the matrix elements by `setElements` method (lines 10-15). The result of both examples looks as follows:

\include SparseMatrixExample_Constructor_init_list_2.out

#### STL map

The constructor which creates the sparse matrix from [`std::map`](https://en.cppreference.com/w/cpp/container/map) is useful especially in situations when you cannot estimate the [matrix row capacities](#setting-of-matrix-row-capacities) in advance. You can first store the matrix elements in [`std::map`](https://en.cppreference.com/w/cpp/container/map) data structure in a [COO](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format manner. It means that each entry of the `map` is the following pair:

```
std::pair( std::pair( row_index, column_index ), element_value )
```

which defines one matrix element at given coordinates `(row_index,column_index)` with given value (`element_value`). Of course, you can insert such entries into the `map` in arbitrary order. When it is complete, you pass the map to the sparse matrix. See the following example:

\includelineno SparseMatrixExample_Constructor_std_map.cpp

The method `setElements` (\ref TNL::Matrices::SparseMatrix::setElements) works the same way for already existing instances of sparse matrix:

\includelineno SparseMatrixExample_setElements_map.cpp

The result of both examples looks as follows:

\include SparseMatrixExample_setElements_map.out

Note, however, that the map can be constructed only on CPU and not on GPU. It requires allocation of additional memory on the host system (CPU) and if the target sparse matrix resided on GPU, the matrix elements must be copied on GPU. This is the reason, why this way of the sparse matrix setup is inefficient compared to other methods.

#### Methods `setElement` and `addElement`

Another way of setting the sparse matrix is by means of the methods `setElement` and `addElement` (\ref TNL::Matrices::SparseMatrix::setElement, \ref TNL::Matrices::SparseMatrix::addElement). The method can be called from both host (CPU) and device (GPU) if the matrix is allocated there. Note, however, that if the matrix is allocated on GPU and the methods are called from CPU there will be significant performance drop because the matrix elements will be transferer one-by-one separately. However, if the matrix elements setup is not a critical part of your algorithm this can be an easy way how to do it. See the following example:

\includelineno SparseMatrixViewExample_setElement.cpp

We first allocate matrix with five rows (it is given by the size of the [initializer list](https://en.cppreference.com/w/cpp/utility/initializer_list) and columns and we set capacity each row to one (line 12). The first for-loop (lines 17-19) runs on CPU no matter where the matrix is allocated. After printing the matrix (lines 21-22), we call the lambda function `f` (lines 24-26) with a help of `ParallelFor` (\ref TNL::Algorithms::ParallelFor , line 28) which is device sensitive and so it runs on CPU or GPU depending on where the matrix is allocated. The result looks as follows:

\include SparseMatrixExample_setElement.out

The method `addElement` (\ref TNL::Matrices::SparseMatrix::addElement) adds a value to specific matrix element. Otherwise, it behaves the same as `setElement`. See the following example:

\includelineno SparseMatrixExample_addElement.cpp

The result looks as follows:

\include SparseMatrixExample_addElement.out

#### Method `getRow`

More efficient method, especially for GPUs, is to combine `getRow` (\ref TNL::Matrices::SparseMatrix::getRow) method with `ParallelFor` (\ref TNL::Algorithms::ParallelFor) and lambda function as the following example demonstrates:

\includelineno SparseMatrixExample_getRow.cpp

On the line 21, we create small matrix having five rows (number of rows is given by the size of the [initializer list](https://en.cppreference.com/w/cpp/utility/initializer_list) ) and columns (number of columns is given by the second parameter) and we set each row capacity to one or three (particular elements of the initializer list). On the line 41, we call `ParallelFor` (\ref TNL::Algorithms::ParallelFor) to iterate over all matrix rows. Each row is processed by the lambda function `f` (lines 24-36). In the lambda function, we first fetch a sparse matrix row (\ref TNL::Matrices::SparseMatrixRowView) (line 25) which serves for accessing particular matrix elements in the matrix row. This object has a method `setElement` (\ref TNL::Matrices::SparseMatrixRowView::setElement) accepting three parameters:

1. `localIdx` is a rank of the nonzero element in given matrix row.
2. `columnIdx` is the new column index of the matrix element.
3. `value` is the new value of the matrix element.

The result looks as follows:

\include SparseMatrixExample_getRow.out

#### Method `forRows`

The method `forRows` (\ref TNL::Matrices::SparseMatrix::forRows) calls the method `getRow` (\ref TNL::Matrices::SparseMatrix::getRow) in parallel. See the following example which has the same effect as the previous one but it is slightly simpler:

\includelineno SparseMatrixExample_forRows.cpp

The differences are:

1. We do not need to get the matrix view as we did in the previous example.
2. We call the method `forAllRows` (\ref TNL::Matrices::SparseMatrix::forAllRows) instead of `ParallelFor` (\ref TNL::Algorithms::ParallelFor) which is simpler since we do not have to state the device type explicitly. The method `forAllRows` calls the method `forRows` for all matrix rows so we do not have to state explicitly the interval of matrix rows neither.
3. The lambda function `f` (lines 27-39) accepts one parameter `row` of the type `RowView` (\ref TNL::Matrices::SparseMatrix::RowView which is \ref TNL::Matrices::SparseMatrixRowView) instead of the index of the matrix row. Therefore we do not need to call the method `getRow` (\ref TNL::Matrices::SparseMatrix::getRow). On the other hand, we need the method `geRowIndex` (\ref TNL::Matrices::SparseMatrixRowView::getRowIndex) to get the index of the matrix row (line 28).

On the lines 46-52, we call a lambda function which computes sum of all elements in a row (lines 47-49) and it divides the row by the `sum` then (lines 50-51).

 The result looks as follows:

\include SparseMatrixExample_forRows.out

#### Method `forElements`

Finally, another efficient way of setting the nonzero matrix elements, is use of the method `forElements` (\ref TNL::Matrices::SparseMatrix::forElements). It requires indexes of the range of rows (`begin` and `end`) to be processed and a lambda function `function` which is called for each nonzero element. The lambda function provides the following data:

* `rowIdx` is a row index of the matrix element.
* `localIdx` is an index of the nonzero matrix element within the matrix row.
* `columnIdx` is a column index of the matrix element. If the matrix element column index is supposed to be modified, this parameter can be a reference and so its value can be changed.
* `value` is a value of the matrix element. If the matrix element value is supposed to be modified, this parameter can be a reference as well and so the element value can be changed.
* `compute` is a bool reference. When it is set to `false` the rest of the row can be omitted. This is, however, only a hint and it depends on the underlying matrix format if it is taken into account.

See the following example:

\includelineno SparseMatrixExample_forElements.cpp

On the line 9, we allocate a lower triangular matrix byt setting the row capacities as `{1,2,3,4,5}`. On the line 11, we prepare lambda function `f` which we execute on the line 22 just by calling the method `forElements` (\ref TNL::Matrices::SparseMatrix::forElements). This method takes the range of matrix rows as the first two parameters and the lambda function as the last parameter. The lambda function receives parameters mentioned above (see the line 11). We first check if the matrix element coordinates (`rowIdx` and `localIdx`) points to an element lying before the matrix diagonal or on the diagonal (line 12). In case of the lower triangular matrix in our example, the local index is in fact the same as the column index

\f[
\left(
\begin{array}{ccccc}
0 & . & . & . & . \\
0 & 1 & . & . & . \\
0 & 1 & 2 & . & . \\
0 & 1 & 2 & 3 & . \\
0 & 1 & 2 & 3 & 4
\end{array}
\right)
\f]

If we call the method `forElements` (\ref TNL::Matrices::SparseMatrix::forElements) to setup the matrix elements for the first time, the parameter `columnIdx` has no sense because the matrix elements and their column indexes were not set yet. Therefore it is important that the test on the line 12 reads as

```
if( rowIdx < localIdx )
```

because

```
if( rowIdx < columnIdx )
```

would not make sense. If we pass through this test, the matrix element lies in the lower triangular part of the matrix and we may set the matrix elements which is done on the lines 17 and 18. The column index (`columnIdx`) is set to local index (line 17) and `value` is set on the line 18. The result looks as follows:

\include SparseMatrixExample_forElements.out

#### Wrapping existing data to sparse matrix view

Standard sparse matrix format like [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) and [Ellpack](https://people.math.sc.edu/Burkardt/data/sparse_ellpack/sparse_ellpack.html) store the matrix elements in specifically defined arrays. In case that you have already allocated them (for example in some other library), they can be wrapped into a sparse matrix view with given matrix format. This can be done by means of functions \ref TNL::Matrices::wrapCSRMatrix and \ref TNL::Matrices::wrapEllpackMatrix . See the following example for demonstration of the CSR format:

\includelineno SparseMatrixViewExample_wrapCSR.cpp

We create sparse matrix having four rows and four columns (line 19). We use TNL vector (\ref TNL::Containers::Vector) to allocate arrays necessary for the CSR format:

1. `valuesVector` (line 20) - contains values of the nonzero matrix elements.
2. `columnIndexesVector` (line 21) - contains column indexes of the nonzero matrix elements.
3. `rowPointersVector` (line 22) - contains positions of the first nonzero matrix elements in each row within `valuesVector` and `columnIndexesVector`. The size of this array equals number of matrix rows plus one.

Next we turn the vectors into C style pointers (lines 24-26) to wrap them into sparse matrix view (line 31). Note, that we must explicitly state the device on which the arrays are allocated. Finlay we print the matrix to check the correctness (line 33). The result looks as follows:

\include SparseMatrixViewExample_wrapCSR.out

Wrapping data corresponding with the Ellpack format is very similar as we can see in the following example:

\includelineno SparseMatrixViewExample_wrapEllpack.cpp

We encode the same sparse matrix as in the previous example. The essence of the Ellpack format is that we allocate the same number of matrix elements for each row which is two in our example. For some matrix rows we use the padding zeros for which we set the column index to -1 (line 21). Therefore the size of `valuesVector` and `columnIndexesVector` equals number of matrix rows times number of matrix elements allocated in each row. As before, we turn the vectors into C style pointers (lines 23-24) and wrap them into sparse matrix view with Ellpack format (line 29). Note that we must state the device on which the arrays are allocated explicitly and also the matrix elements organization, which is \ref TNL::Algorithms::Segments::RowMajorOrder in this case. For Ellpack matrix stored on GPU, \ref TNL::Algorithms::Segments::ColumnMajorOrder is preferred. The result looks as follows:

\include SparseMatrixViewExample_wrapEllpack.out

#### Symmetric sparse matrices

For sparse [symmetric matrices](https://en.wikipedia.org/wiki/Symmetric_matrix), TNL offers a format storing only a half of the matrix elements. More precisely, ony the matrix diagonal and the elements bellow are stored in the memory. The matrix elements above the diagonal are deduced from those bellow. If such a symmetric format is used on GPU, atomic operations must be used in some matrix operations. For this reason, symmetric matrices can be combined only with matrix elements values expressed in `float` or `double` type. An advantage of the symmetric formats is lower memory consumption. Since less data need to be transferred from the memory, better performance might be observed. In some cases, however, the use of atomic operations on GPU may cause performance drop. Mostly we can see approximately the same performance compared to general formats but we can profit from lower memory requirements which is appreciated especially on GPU. The following example shows how to create symmetric sparse matrix.

\includelineno SymmetricSparseMatrixExample.cpp

We construct matrix of the following form

\f[
\left(
\begin{array}{ccccc}
 1  & \color{grey}{2} & \color{grey}{3} & \color{grey}{4} & \color{grey}{5}  \\
 2  &  1 &    &    &     \\
 3  &    &  1 &    &     \\
 4  &    &    &  1 &     \\
 5  &    &    &    &  1
\end{array}
\right)
\f]

The elements depicted in grey color are not stored in the memory. The main difference, compared to creation of general sparse matrix, is on line 9 where we state that the matrix is symmetric by setting the matrix type to \ref TNL::Matrices::SymmetricMatrix. Next we set only the diagonal elements and those lying bellow the diagonal (lines 13-17). When we print the matrix (line 19) we can see also the symmetric part above the diagonal. Next we test product of matrix and vector (lines 21-23). The result looks as follows:

\include SymmetricSparseMatrixExample.out

**Warning: Assignment of symmetric sparse matrix to general sparse matrix does not give correct result, currently. Only the diagonal and the lower part of the matrix is assigned.**

#### Binary sparse matrices

If the matrix element value type (i.e. `Real` type) is set to `bool` the matrix elements can be only `1` or `0`. So in the sparse matrix formats, where we do not store the zero matrix elements, explicitly stored elements can have only one possible value which is `1`.  Therefore we do not need to store the values, only the positions of the nonzero elements. The array `values`, which usualy stores the matrix elements values, can be completely omitted and we can reduce the memory requirements. The following table shows how much we can reduce the memory consumption when using binary matrix instead of common sparse matrix using `float` or `double` types:

| Real   | Index  | Common sparse matrix | Binary sparse matrix | Ratio      |
|:------:|:------:|:--------------------:|:--------------------:|:----------:|
| float  | 32-bit |         4 + 4 =  8 B |                  4 B |        50% |
| float  | 64-bit |         4 + 8 = 12 B |                  8 B |        75% |
| double | 32-bit |         8 + 4 = 12 B |                  4 B |        33% |
| double | 64-bit |         8 + 8 = 16 B |                  8 B |        50% |

The following example demonstrates the use of binary matrix:

\includelineno BinarySparseMatrixExample.cpp

All we need to do is set the `Real` type to `bool` as we can see on the line 9. We can see that even though we set different values to different matrix elements (lines 14-18) at the end all of them are turned into ones (printing of the matrix on the line 20). There is an issue, however, which is demonstrated on the product of the matrix with a vector. Nonbinary matrices compute all operations using the `Real` type. If it is set to `bool` operations like [SpMV](https://en.wikipedia.org/wiki/Sparse_matrix-vector_multiplication) would not get correct solution. Therefore sparse matrices use another type called `ComputeReal` which is the 6th template parameter of \ref TNL::Matrices::SparseMatrix. By default it is set to `Index` type but it can be changed by the user. On the lines 26-29 we show how to change this type to `double` and what is the effect of it (correct result of matrix-vector multiplication). The result looks as follows:

\include BinarySparseMatrixExample.out

### Tridiagonal matrices

Tridiagonal matrix format serves for specific matrix pattern when the nonzero matrix elements can be placed only at the diagonal and immediately next to the diagonal. Here is an example:

\f[
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2
 \end{array}
 \right)
\f]

An advantage is that we do not store the column indexes explicitly as it is in \ref TNL::Matrices::SparseMatrix. This can reduce significantly the  memory requirements which also means better performance. See the following table for the storage requirements comparison between \ref TNL::Matrices::TridiagonalMatrix and \ref TNL::Matrices::SparseMatrix.

  Real   | Index      |      SparseMatrix    | TridiagonalMatrix   | Ratio
 --------|------------|----------------------|---------------------|--------
  float  | 32-bit int | 8 bytes per element  | 4 bytes per element | 50%
  double | 32-bit int | 12 bytes per element | 8 bytes per element | 75%
  float  | 64-bit int | 12 bytes per element | 4 bytes per element | 30%
  double | 64-bit int | 16 bytes per element | 8 bytes per element | 50%

Tridiagonal matrix is a templated class defined in the namespace \ref TNL::Matrices. It has five template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for GPU supporting CUDA. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.

For better alignment in the memory the tridiagonal format is organized like if there were three nonzero matrix elements in each row. This is not true for example in the first row where there is no matrix element on the left side of the diagonal. The same happens on the last row of the matrix. We have to add even the artificial matrix elements like this:

\f[
\begin{array}{c}
0 \\
. \\
. \\
. \\
. \\
.
\end{array}
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2
 \end{array}
 \right)
 \begin{array}{c}
. \\
. \\
. \\
. \\
. \\
0
\end{array}
\f]

If the tridiagonal matrix has more rows then columns, we have to extend the last two rows with nonzero elements in this way

\f[
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2 \\
  .  &  .  &  .  &  .  &  . & -1
 \end{array}
 \right)
\rightarrow
\begin{array}{c}
0 \\
. \\
. \\
. \\
. \\
. \\
.
\end{array}
\left(
 \begin{array}{ccccccc}
  2  & -1  &  .  & .   &  . & .  \\
 -1  &  2  & -1  &  .  &  . & .  \\
  .  & -1  &  2  & -1  &  . & .  \\
  .  &  .  & -1  &  2  & -1 &  . \\
  .  &  .  &  .  & -1  &  2 & -1 \\
  .  &  .  &  .  &  .  & -1 &  2 \\
  .  &  .  &  .  &  .  &  . & -1
 \end{array}
 \right)
 \begin{array}{cc}
. & . \\
. & . \\
. & . \\
. & . \\
. & . \\
0 & . \\
0 & 0
\end{array}
\f]

We also would like to remind the meaning of the local index (`localIdx`) of the matrix element within a matrix row. It is a rank of the nonzero matrix element in given row as we explained  in section [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices). The values of the local index for tridiagonal matrix elements are as follows

\f[
\left(
\begin{array}{cccccc}
1 & 2 &   &   &   &     \\
0 & 1 & 2 &   &   &     \\
  & 0 & 1 & 2 &   &     \\
  &   & 0 & 1 & 2 &     \\
  &   &   & 0 & 1 & 2   \\
  &   &   &   & 0 & 1
\end{array}
\right)
\f]


In the following text we show different methods for setup of tridiagonal matrices.

#### Initializer list

The tridiagonal matrix can be initialized by the means of the constructor with [initializer list](https://en.cppreference.com/w/cpp/utility/initializer_list). The matrix from the beginning of this section can be constructed as the following example demonstrates:

\includelineno TridiagonalMatrixExample_Constructor_init_list_1.cpp

The matrix elements values are defined on lines 39-44. Each matrix row is represented by embedded an initializer list. We set three values in each row including the padding zeros.

The output of the example looks as:

\include TridiagonalMatrixExample_Constructor_init_list_1.out

#### Methods setElement and addElement

Similar way of the tridiagonal matrix setup is offered by the method `setElements` (\ref TNL::Matrices::TridiagonalMatrix::setElements) as the following example demonstrates:

\includelineno TridiagonalMatrixExample_setElements.cpp

Here we create the matrix in two steps. Firstly, we setup the matrix dimensions by the appropriate constructor (line 24) and after that we setup the matrix elements (line 25-45). The result looks the same as in the previous example:

\include TridiagonalMatrixExample_setElements.out

In the following example we create tridiagonal matrix with 5 rows and 5 columns (line 12-14) by the means of a shared pointer (\ref TNL::Pointers::SharedPointer) to make this work even on GPU. We set numbers 0,...,4 on the diagonal (line 16) and we print the matrix (line 18). Next we use a lambda function (lines 21-27) combined with parallel for (\ref TNL::Algorithms::ParallelFor) (line 35), to modify the matrix. The offdiagonal elements are set to 1 (lines 23 and 26) and for the diagonal elements, we change the sign (line 24).

\includelineno TridiagonalMatrixExample_setElement.cpp

The result looks as follows:

\include TridiagonalMatrixExample_setElement.out

#### Method getRow

 A bit different way of setting up the matrix, is the use of tridiagonal matrix view and the method `getRow` (\ref TNL::Matrices::TridiagonalMatrixView::getRow) as the following example demonstrates:

\includelineno TridiagonalMatrixViewExample_getRow.cpp

We create a matrix with the same size (line 22-27). Next, we fetch the tridiagonal matrix view (ef TNL::Matrices::TridiagonalMatrixView ,line 28) which we use in the lambda function for matrix elements modification (lines 30-38). Inside the lambda function, we first get a matrix row by calling the method `getRow` (\ref TNL::Matrices::TridiagonalMatrixView::getRow) using which we can access the matrix elements (lines 33-37). We would like to stress that the method `setElement` addresses the matrix elements with the `localIdx` parameter which is a rank of the nonzero element in the matrix row - see [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices). The lambda function is called by the `ParallelFor` (\ref TNL::Algorithms::ParallelFor).

The result looks as follows:

\include TridiagonalMatrixViewExample_getRow.out

#### Method forRows

As in the case of other matrix types, the method `forRows` (\ref TNL::Matrices::TridiagonalMatrix::forRows) calls the method `getRow` (\ref TNL::Matrices::TridiagonalMatrix::getRow) in parallel. It is demonstrated by the following example which we may directly compare with the previous one:

\includelineno TridiagonalMatrixExample_forRows.cpp

The differences are:

1. We do not need to get the matrix view as we did in the previous example.
2. We call the method `forAllRows` (\ref TNL::Matrices::TridiagonalMatrix::forAllRows) (line 33) instead of `ParallelFor` (\ref TNL::Algorithms::ParallelFor) which is simpler since we do not have to state the device type explicitly. The method `forAllRows` calls the method `forRows` for all matrix rows so we do not have to state explicitly the interval of matrix rows neither.
3. The lambda function `f` (lines 25-31) accepts one parameter `row` of the type `RowView` (\ref TNL::Matrices::TridiagonalMatrix::RowView which is \ref TNL::Matrices::TridiagonalMatrixRowView) instead of the index of the matrix row. Therefore we do not need to call the method `getRow` (\ref TNL::Matrices::TridiagonalMatrix::getRow). On the other hand, we need the method `geRowIndex` (\ref TNL::Matrices::TridiagonalMatrixRowView::getRowIndex) to get the index of the matrix row (line 24).

Next, we compute sum of absolute values of matrix elements in each row and store it in a vector (lines 39-46). Firstly we create the vector `sum_vector` for storing the sums (line 39) and get a vector view `sum_view` to get access to the vector from a lambda function. On the lines 41-46, we call lambda function for each matrix row which iterates over all matrix elements and sum their absolute values. Finally we store the result to the output vector (line 45).

The result looks as follows:

\include TridiagonalMatrixExample_forRows.out

#### Method forElements

Finally, even a bit more simple way of matrix elements manipulation with the method `forElements` (\ref TNL::Matrices::TridiagonalMatrix::forElements) is demonstrated in the following example:

\includelineno TridiagonalMatrixViewExample_forElements.cpp

On the line 41, we call the method `forElements` (\ref TNL::Matrices::TridiagonalMatrix::forElements) instead of parallel for (\ref TNL::Algorithms::ParallelFor). This method iterates over all matrix rows and all nonzero matrix elements. The lambda function on the line 24 therefore do not receive only the matrix row index but also local index of the matrix element (`localIdx`) which is a rank of the nonzero matrix element in given row  - see [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices). Next parameter, `columnIdx` received by the lambda function, is the column index of the matrix element. The fourth parameter `value` is a reference on the matrix element which we use for its modification. If the last parameter `compute` is set to false, the iterations over the matrix rows is terminated.

The result looks as follows:

\include TridiagonalMatrixViewExample_forElements.out

### Multidiagonal matrices

Multidiagonal matrices are generalization of the tridiagonal ones. It is a special type of sparse matrices with specific pattern of the nonzero matrix elements which are positioned only parallel along diagonal. See the following example:

\f[
  \left(
  \begin{array}{ccccccc}
   4  & -1  &  .  & -1  &  . & .  \\
  -1  &  4  & -1  &  .  & -1 & .  \\
   .  & -1  &  4  & -1  &  . & -1 \\
  -1  & .   & -1  &  4  & -1 &  . \\
   .  & -1  &  .  & -1  &  4 & -1 \\
   .  &  .  & -1  &  .  & -1 &  4
  \end{array}
  \right)
 \f]

 We can see that the matrix elements lay on lines parallel to the main diagonal. Such lines can be characterized by their offsets from the main diagonal. On the following figure, each such line is depicted in different color:

  \f[
\begin{array}{ccc}
\color{green}{-3} & .                & \color{cyan}{-1} \\
\hline
 \color{green}{*} & .                & \color{cyan}{*} \\
 .                & \color{green}{*} & . \\
 .                & .                & \color{green}{*} \\
 .                & .                & . \\
 .                & .                & . \\
 .                & .                & .
\end{array}
\left(
  \begin{array}{ccccccc}
 \color{blue}{0}    & \color{magenta}{1}   & .                   & \color{red}{3}      & .                   & . \\
   \hline
  \color{blue}{4}   & \color{magenta}{-1}  &  .                  & \color{red}{-1}     &  .                  & .  \\
  \color{cyan}{-1}  & \color{blue}{4}      & \color{magenta}{-1} &  .                  & \color{red}{-1}     & .  \\
   .                & \color{cyan}{-1}     & \color{blue}{4}     & \color{magenta}{-1} &  .                  & \color{red}{-1} \\
  \color{green}{-1} & .                    & \color{cyan}{-1}    & \color{blue}{4}     & \color{magenta}{-1} &  . \\
   .                & \color{green}{-1}    &  .                  & \color{cyan}{-1}    &  \color{blue}{4}    & \color{magenta}{-1} \\
   .                &  .                   & \color{green}{-1}   &  .                  & \color{cyan}{-1}    &  \color{blue}{4}
  \end{array}
  \right)
 \f]

 In this matrix, the offsets reads as \f$\{-3, -1, 0, +1, +3\}\f$. It also means that the column indexes on \f$i-\f$th row are \f$\{i-3, i-1, i, i+1, i+3\}\f$ (where we accept only nonnegative indexes smaller than the number of matrix columns). An advantage is that, similar to the tridiagonal matrix (\ref TNL::Matrices::TridiagonalMatrix), we do not store the column indexes explicitly as it is in \ref TNL::Matrices::SparseMatrix. This can significantly reduce the  memory requirements which also means better performance. See the following table for the storage requirements comparison between multidiagonal matrix (\ref TNL::Matrices::MultidiagonalMatrix) and general sparse matrix (\ref TNL::Matrices::SparseMatrix).

  Real   | Index     |      SparseMatrix    | MultidiagonalMatrix | Ratio
 --------|-----------|----------------------|---------------------|--------
  float  | 32-bit int| 8 bytes per element  | 4 bytes per element | 50%
  double | 32-bit int| 12 bytes per element | 8 bytes per element | 75%
  float  | 64-bit int| 12 bytes per element | 4 bytes per element | 30%
  double | 64-bit int| 16 bytes per element | 8 bytes per element | 50%

 For the sake of better memory alignment and faster access to the matrix elements, we store all subdiagonals in complete form including the elements which are outside the matrix as depicted on the following figure where zeros stand for the padding artificial zero matrix elements

\f[
\begin{array}{cccc}
0  &   &   & 0  \\
   & 0 &   &    \\
   &   & 0 &    \\
   &   &   & 0  \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &
\end{array}
\left(
\begin{array}{cccccccccccccccc}
1  &  0 &    &    &  0 &    &    &    &    &    &    &    &     &    &    &   \\
0  &  1 &  0 &    &    &  0 &    &    &    &    &    &    &     &    &    &   \\
   &  0 &  1 &  0 &    &    & 0  &    &    &    &    &    &     &    &    &   \\
   &    &  0 &  1 &  0 &    &    &  0 &    &    &    &    &     &    &    &   \\
0  &    &    &  0 &  1 & 0  &    &    & 0  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & 1  & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 &  1 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & 0  &    &    &  0 &  1 & 0  &    &    & 0  &     &    &    &   \\
   &    &    &    & 0  &    &    &  0 & 1  &  0 &    &    &  0  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 &  1 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 &  1 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  0 &    &    &  0 &  1 &  0  &    &    & 0 \\
   &    &    &    &    &    &    &    & 0  &    &    &  0 &  1  & 0  &    &   \\
   &    &    &    &    &    &    &    &    &  0 &    &    &  0  & 1  & 0  &   \\
   &    &    &    &    &    &    &    &    &    &  0 &    &     & 0  & 1  & 0 \\
   &    &    &    &    &    &    &    &    &    &    & 0  &     &    & 0  & 1
\end{array}
\right)
\begin{array}
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
0  &   &   &    \\
   & 0 &   &    \\
   &   & 0 &    \\
0  &   &   & 0
\end{array}
\f]

Multidiagonal matrix is a templated class defined in the namespace \ref TNL::Matrices. It has six template parameters:

* `Real` is a type of the matrix elements. It is `double` by default.
* `Device` is a device where the matrix shall be allocated. Currently it can be either \ref TNL::Devices::Host for CPU or \ref TNL::Devices::Cuda for CUDA supporting GPUs. It is \ref TNL::Devices::Host by default.
* `Index` is a type to be used for indexing of the matrix elements. It is `int` by default.
* `ElementsOrganization` defines the organization of the matrix elements in memory. It can be \ref TNL::Algorithms::Segments::ColumnMajorOrder or \ref TNL::Algorithms::Segments::RowMajorOrder for column-major and row-major organization respectively. Be default, it is the row-major order if the matrix is allocated in the host system and column major order if it is allocated on GPU.
* `RealAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements. By default, it is the default allocator for given `Real` type and `Device` type -- see \ref TNL::Allocators::Default.
* `IndexAllocator` is a memory allocator (one from \ref TNL::Allocators) which shall be used for allocation of the matrix elements offsets. By default, it is the default allocator for given `Index` type and `Device` type -- see \ref TNL::Allocators::Default.

In the following text we show different methods how to setup multidiagonal matrices.

#### Initializer list

Smaller multidiagonal matrices can be created using the constructor with initializer lists (\ref std::initializer_list) as we demonstrate in the following example:

\includelineno MultidiagonalMatrixExample_Constructor_init_list_2.cpp

Here, we create a matrix which looks as

\f[
\left(
\begin{array}{cccccc}
4  & -1 &    & -1 &    &    \\
-1 &  4 & -1 &    & -1 &    \\
   & -1 & 4  & -1 &    & -1 \\
-1 &    & -1 &  4 & -1 &    \\
   & -1 &    & -1 & 4  & -1 \\
   &    & -1 &    & -1 &  4 \\
\end{array}
\right).
\f]

On the lines 25-46, we call the constructor which, in addition to matrix dimensions and subdiagonals offsets, accepts also initializer list of initializer lists with matrix elements values. Each embedded list corresponds to one matrix row and it contains values of matrix elements on particular subdiagonals including those which lies out of the matrix. The result looks as follows:

\include MultidiagonalMatrixExample_Constructor_init_list_2.out

#### Methods setElement and addElement

Another and more efficient way of setting the matrix elements is by means of the method `setElement` (\ref TNL::Matrices::MultidiagonalMatrix::setElement). It is demonstrated in the following example:

\includelineno MultidiagonalMatrixViewExample_setElement.cpp

This examples shows that the method `setElement` can be used both on the host (CPU) (line 19) as well as in the lambda functions that can be called from GPU kernels (lines 25-29). For this purpose, we fetch a matrix view on the line 16. The result looks as follows:

\include MultidiagonalMatrixViewExample_setElement.out

#### Method getRow

Slightly more efficient way of the multidiagonal matrix setup is offered by the method `getRow` (\ref TNL::Matrices::MultidiagonalMatrix::getRow). We will use it to create a matrix of the following form:

\f[
\left(
\begin{array}{cccccccccccccccc}
1  &  . &    &    &  . &    &    &    &    &    &    &    &     &    &    &   \\
.  &  1 &  . &    &    &  . &    &    &    &    &    &    &     &    &    &   \\
   &  . &  1 &  . &    &    & .  &    &    &    &    &    &     &    &    &   \\
   &    &  . &  1 &  . &    &    &  . &    &    &    &    &     &    &    &   \\
.  &    &    &  . &  1 & .  &    &    & .  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & -4 & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 & -4 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & .  &    &    &  . &  1 & .  &    &    & .  &     &    &    &   \\
   &    &    &    & .  &    &    &  . & 1  &  . &    &    &  .  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 & -4 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 & -4 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  . &    &    &  . &  1 &  .  &    &    & . \\
   &    &    &    &    &    &    &    & .  &    &    &  . &  1  & .  &    &   \\
   &    &    &    &    &    &    &    &    &  . &    &    &  .  & 1  & .  &   \\
   &    &    &    &    &    &    &    &    &    &  . &    &     & .  & 1  & . \\
   &    &    &    &    &    &    &    &    &    &    & .  &     &    & .  & 1
\end{array}
\right)
\f]

The matrices of this form arise from a discretization of the [Laplace operator in 2D](https://en.wikipedia.org/wiki/Discrete_Laplace_operator) by the [finite difference method](https://en.wikipedia.org/wiki/Discrete_Poisson_equation). We use this example because it is a frequent numerical problem and the multidiagonal format is very suitable for such matrices. If the reader, however, is not familiar with the finite difference method, please, do not be scared, we will just create the matrix mentioned above. The code based on use of method `getRow` reads as:

\includelineno MultidiagonalMatrixExample_Constructor.cpp

We firstly compute the matrix size (`matrixSize`) based on the numerical grid dimensions on the line 16. The subdiagonals offsets are defined by the numerical grid size and since it is four in this example the offsets read as \f$\left\{-4,-1,0,1,4 \right\} \f$ or `{ -gridSize, -1, 0, 1, gridSize}` (line 17). Here we store the offsets in vector (\ref TNL::Containers::Vector) called `offsets`. Next we use a constructor with matrix dimensions and offsets passed via TNL vector (line 18) and we fetch a matrix view (\ref TNL::Matrices::MultidiagonalMatrixView, line 19).

The matrix is constructed by iterating over particular nodes of the numerical grid. Each node correspond to one matrix row. This is why the lambda function `f` (lines 20-35) take two indexes `i` and `j` (line 20). Their values are coordinates of the two-dimensional numerical grid. Based on these coordinates we compute index (`elementIdx`) of the corresponding matrix row (line 21). We fetch matrix row (`row`) by calling the `getRow` method (\ref TNL::Matrices::MultidiagonalMatrix::getRow) (line 22). Depending on the grid node coordinates we set either the boundary conditions (lines 23-26) for the boundary nodes (those laying on the boundary of the grid and so their coordinates fulfil the condition `i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1` ) for which se set only the diagonal element to 1. The inner nodes of the numerical grid are handled on the lines 29-33 where we set coefficients approximating the Laplace operator. We use the method `setElement` of the matrix row (\ref TNL::Matrices::MultidiagonalMatrixRow::setElement) which takes the local index of the nonzero matrix element as the first parameter (see [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices)) and the new value of the element as the second parameter. The local indexes, in fact, refer to particular subdiagonals as depicted on the following figure (in blue):

\f[
\begin{array}{cccc}
\color{blue}{-4} &   &   & \color{blue}{-1} \\
\hline
.  &   &   & .  \\
   & . &   &    \\
   &   & . &    \\
   &   &   & .  \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &    \\
   &   &   &
\end{array}
\left(
\begin{array}{cccccccccccccccc}
\color{blue}{0}  &  \color{blue}{1} &    &    &  \color{blue}{4} &    &    &    &    &    &    &    &     &    &    &   \\
\hline
1  &  . &    &    &  . &    &    &    &    &    &    &    &     &    &    &   \\
.  &  1 &  . &    &    &  . &    &    &    &    &    &    &     &    &    &   \\
   &  . &  1 &  . &    &    & .  &    &    &    &    &    &     &    &    &   \\
   &    &  . &  1 &  . &    &    &  . &    &    &    &    &     &    &    &   \\
.  &    &    &  . &  1 & .  &    &    & .  &    &    &    &     &    &    &   \\
   & -1 &    &    & -1 & 1  & -1 &    &    & -1 &    &    &     &    &    &   \\
   &    & -1 &    &    & -1 &  1 & -1 &    &    & -1 &    &     &    &    &   \\
   &    &    & .  &    &    &  . &  1 & .  &    &    & .  &     &    &    &   \\
   &    &    &    & .  &    &    &  . & 1  &  . &    &    &  .  &    &    &   \\
   &    &    &    &    & -1 &    &    & -1 &  1 & -1 &    &     & -1 &    &   \\
   &    &    &    &    &    & -1 &    &    & -1 &  1 & -1 &     &    & -1 &   \\
   &    &    &    &    &    &    &  . &    &    &  . &  1 &  .  &    &    & . \\
   &    &    &    &    &    &    &    & .  &    &    &  . &  1  & .  &    &   \\
   &    &    &    &    &    &    &    &    &  . &    &    &  .  & 1  & .  &   \\
   &    &    &    &    &    &    &    &    &    &  . &    &     & .  & 1  & . \\
   &    &    &    &    &    &    &    &    &    &    & .  &     &    & .  & 1
\end{array}
\right)
\f]

We use `ParallelFor2D` (\ref TNL::Algorithms::ParallelFor2D) to iterate over all nodes of the numerical grid (line 36) and apply the lambda function. The result looks as follows:

\include MultidiagonalMatrixExample_Constructor.out

#### Method forRows

As in the case of other matrix types, the method `forRows` (\ref TNL::Matrices::MultidiagonalMatrix::forRows) calls the method `getRow` (\ref TNL::Matrices::MultidiagonalMatrix::getRow) in parallel. It is demonstrated by the following example:

\includelineno MultidiagonalMatrixExample_forRows.cpp

 We call the method `forAllRows` (\ref TNL::Matrices::MultidiagonalMatrix::forAllRows) (line 36) instead of `ParallelFor` (\ref TNL::Algorithms::ParallelFor) which is simpler since we do not have to state the device type explicitly. The method `forAllRows` calls the method `forRows` for all matrix rows so we do not have to state explicitly the interval of matrix rows neither. The lambda function `f` (lines 28-35) accepts one parameter `row` of the type `RowView` (\ref TNL::Matrices::MultidiagonalMatrix::RowView which is \ref TNL::Matrices::MultidiagonalMatrixRowView). At the beginning of the lambda function, we call the method `geRowIndex` (\ref TNL::Matrices::MultidiagonalMatrixRowView::getRowIndex) to get the index of the matrix row (line 29).

Next, we compute sum of absolute values of matrix elements in each row and store it in a vector (lines 39-46). Firstly we create the vector `sum_vector` for storing the sums (line 39) and get a vector view `sum_view` to get access to the vector from a lambda function. On the lines 41-46, we call lambda function for each matrix row which iterates over all matrix elements and sum their absolute values. Finally we store the result to the output vector (line 45).

The result looks as follows:

\include MultidiagonalMatrixExample_forRows.out

#### Method forElements

Similar and even a bit simpler way of setting the matrix elements is offered by the method `forElements` (\ref TNL::Matrices::MultidiagonalMatrix::forElements, \ref TNL::Matrices::MultidiagonalMatrixView::forElements) as demonstrated in the following example:

\includelineno MultidiagonalMatrixViewExample_forElements.cpp

In this case, we need to provide a lambda function `f` (lines 27-43) which is called for each matrix row just by the method `forElements` (line 44). The lambda function `f` provides the following parameters

* `rowIdx` is an index iof the matrix row.
* `localIdx` is in index of the matrix subdiagonal.
* `columnIdx` is a column index of the matrix element.
* `value` is a reference to the matrix element value. It can be used even for changing the value.
* `compute` is a reference to boolean. If it is set to false, the iteration over the matrix row can be stopped.

In this example, the matrix element value depends only on the subdiagonal index `localIdx` (see [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices)) as we can see on the line 42. The result looks as follows:

\include MultidiagonalMatrixExample_forElements.out

### Lambda matrices

Lambda matrix (\ref TNL::Matrices::LambdaMatrix) is a special type of matrix which could be also called *matrix-free matrix*. The matrix elements are not stored in memory explicitly but they are evaluated **on-the-fly** by means of user defined lambda functions. If the matrix elements can be expressed by computationally not expansive formula, we can significantly reduce the memory consumption which is appreciated especially on GPUs. Since the memory accesses are quite expensive even on both CPU and GPU, we can get, at the end, even much faster code.

The lambda matrix (\ref TNL::Matrices::LambdaMatrix) is a templated class with the following template parameters:

* `MatrixElementsLambda` is a lambda function which evaluates the matrix elements values and column indexes.
* `CompressedRowLengthsLambda` is a lambda function telling how many nonzero elements are there in given matrix row.
* `Real` is a type of matrix elements values.
* `Device` is a device on which the lambda functions mentioned above will be evaluated.
* `Index` is a type to be used for indexing.

The lambda function `MatrixElementsLambda` is supposed to have the following declaration:

\includelineno snippet_MatrixElementsLambda_declaration.cpp

where the particular parameters have the following meaning:

* `rows` tells the number of matrix rows.
* `columns` tells the number of matrix columns.
* `rowIdx` is index of the matrix row in which we are supposed to evaluate the matrix element.
* `localIdx` is a rank of the nonzero matrix element, see [Indexing of nonzero matrix elements in sparse matrices](#indexing-of-nonzero-matrix-elements-in-sparse-matrices).
* `columnIdx` is a reference on variable where we are supposed to store the matrix element column index.
* `value` is a reference on variable where we are supposed to store the matrix element value.

The lambda function `CompressedRowLengthsLambda` (by compressed row length we mean the number of matrix elements in a row after ignoring/compressing the zero elements) is supposed to be declared like this:

\includelineno snippet_CompressedRowLengthsLambda_declaration.cpp

where the parameters can be described as follows:

* `rows` tells the number of matrix rows.
* `columns` tells the number of matrix columns.
* `rowIdx` is index of the matrix row for which we are supposed to evaluate the number of nonzero matrix elements.

The lambda function is supposed to return just the number of the nonzero matrix elements in given matrix row.

#### Lambda matrix inititation

How to put the lambda functions together with the lambda matrix is demonstrated in the following example:

\includelineno LambdaMatrixExample_Constructor.cpp

Here we create two simple diagonal matrices. Therefore they share the same lambda function `compressedRowLengths` telling the number of nonzero matrix elements in particular matrix rows which is always one (line 9). The first matrix, defined by the lambda function `matrixElements1`, is identity matrix and so its each diagonal element equals one. We set the matrix element value to `1.0` (line 12) and the column index equals the row index (line 15). The second matrix, defined by the lambda function `matrixElements2`, is also diagonal but not the identity matrix. The values of the diagonal elements equal to row index (line 16).

With the same lambda functions we can define matrices with different dimensions. In this example, we set the matrix size to five (line 19). It could be quite difficult to express the lambda matrix type because it depends on the types of the lambda functions. To make this easier, one may use the lambda-matrix factory (\ref TNL::Matrices::LambdaMatrixFactory). Using `decltype` one can deduce even the matrix type (line 24) followed by calling lambda matrix constructor with matrix dimensions and instances of the lambda functions (line 25). Or one can just simply employ the keyword `auto` (line 30) followed by setting the matrix dimensions (line 31).

The result looks as follows:

\include LambdaMatrixExample_Constructor.out

#### Method forRows

Method `forRows` (\ref TNL::Matrices::LambdaMatrix::forRows, \ref TNL::Matrices::LambdaMatrix::forAllRows) iterates in parallel over all matrix rows. In the case of lambda matrices, it cannot be used for changing the matrix elements since they cannot be changed. In the following example, we show how to use this method to copy the matrix elements values to the dense matrix:

\includelineno LambdaMatrixExample_forRows.cpp

We start with the lambda functions (lines 17-61) defining the elements of the lambda matrix. Next, we create the lambda matrix `matrix` (lines 62-64) and the dense matrix `denseMatrix` (lines 67-68) together with the dense matrix view (line 69). The lambda function `f` (lines 70-74) serves for copying matrix elements from the lambda matrix to the dense matrix. The process of matrix elements copying is started by calling the method `forAllRows` (\ref TNL::Matrices::LambdaMatrix::forRows, \ref TNL::Matrices::LambdaMatrix::forAllRows) (line 75).

Note, however, that use of `forElements` method (\ref TNL::Matrices::LambdaMatrix::forElements) would be more convenient.

Next, we compute sum of absolute values of matrix elements in each row and store it in a vector (lines 83-90). Firstly we create the vector `sum_vector` for storing the sums (line 83) and get a vector view `sum_view` to get access to the vector from a lambda function. On the lines 85-90, we call lambda function for each matrix row which iterates over all matrix elements and sum their absolute values. Finally we store the result to the output vector (line 92).



The result looks as follows:

\include LambdaMatrixExample_forRows.out

#### Method forElements

The lambda matrix has the same interface as other matrix types except of the method `getRow`. The following example demonstrates the use of the method `forElements` (\ref TNL::Matrices::LambdaMatrix::forElements) to copy the lambda matrix into the dense matrix:

\includelineno LambdaMatrixExample_forElements.cpp

Here, we treat the lambda matrix as if it was dense matrix and so the lambda function `compressedRowLengths` returns the number of the nonzero elements equal to the number of matrix columns (line 13). However, the lambda function `matrixElements` (lines 14-17), sets nonzero values only to lower triangular part of the matrix. The elements in the upper part are equal to zero (line 16). Next we create an instance of the lambda matrix with a help of the lambda matrix factory (\ref TNL::Matrices::LambdaMatrixFactory) (lines 19-20) and an instance of the dense matrix (\ref TNL::Matrices::DenseMatrix) (lines 22-23).

Next we call the lambda function `f` by the method `forElements` (\ref TNL::Matrices::LambdaMatrix::forElements) to set the matrix elements of the dense matrix `denseMatrix` (line 26) via the dense matrix view (`denseView`) (\ref TNL::Matrices::DenseMatrixView). Note, that in the lambda function `f` we get the matrix element value already evaluated in the variable `value` as we are used to from other matrix types. So in fact, the same lambda function `f` would do the same job even for sparse matrix or any other. Also note, that in this case we iterate even over all zero matrix elements because the lambda function `compressedRowLengths` (line 13) tells so. The result looks as follows:

\include LambdaMatrixExample_forElements.out

At the end of this part, we show two more examples, how to express a matrix approximating the Laplace operator:

\includelineno LambdaMatrixExample_Laplace.cpp

The following is another way of doing the same but with precomputed supporting vectors:

\includelineno LambdaMatrixExample_Laplace_2.cpp

The result of both examples looks as follows:

\include LambdaMatrixExample_Laplace.out

### Distributed matrices

TODO: Write documentation on distributed matrices.

## Flexible reduction in matrix rows

Flexible reduction in matrix rows is a powerful tool for many different matrix operations. It is represented by the method `reduceRows` (\ref TNL::Matrices::DenseMatrix::reduceRows,
\ref TNL::Matrices::SparseMatrix::reduceRows, \ref TNL::Matrices::TridiagonalMatrix::reduceRows, \ref TNL::Matrices::MultidiagonalMatrix::reduceRows, \ref TNL::Matrices::LambdaMatrix::reduceRows) and similar to the method `forElements` it iterates over particular matrix rows. However, it performs *flexible paralell reduction* in addition. For example, the matrix-vector product can be seen as a reduction of products of matrix elements with the input vector in particular matrix rows. The first element of the result vector ios obtained as:

\f[
y_1 = a_{11} x_1 + a_{12} x_2 + \ldots + a_{1n} x_n = \sum_{j=1}^n a_{1j}x_j
\f]

and in general i-th element of the result vector is computed as

\f[
y_i = a_{i1} x_1 + a_{i2} x_2 + \ldots + a_{in} x_n = \sum_{j=1}^n a_{ij}x_j.
\f]

We see that in i-th matrix row we have to compute the sum \f$\sum_{j=1}^n a_{ij}x_j\f$ which is reduction of products \f$ a_{ij}x_j\f$. Similar to flexible parallel reduction (\ref TNL::Algorithms::Reduction) we just need to design proper lambda functions. There are three of them.

1. `fetch` reads and preprocesses data entering the flexible parallel reduction.
2. `reduce` performs the reduction operation.
3. `keep` stores the results from each matrix row.

#### Lambda function fetch

This lambda function has the same purpose as the lambda function `fetch` in flexible parallel reduction for arrays and vectors (see [Flexible Parallel Reduction](../ReductionAndScan/tutorial_ReductionAndScan.md)). It is supposed to be declared as follows:

\includelineno snippet_rows_reduction_fetch_declaration.cpp

The meaning of the particular parameters is as follows:

1. `rowIdx` is the row index of the matrix element.
2. `columnIdx` is the column index of the matrix element.
3. `value` is the value of the matrix element.

The lambda function returns a value of type `Real` based on the input data.

#### Lambda function reduce

The lambda function `reduce` expresses reduction operation (sum, product, minimum, maximum etc.) which is supposed to be done during the flexible reduction.

\includelineno snippet_rows_reduction_reduce_declaration.cpp

The meaning of the particular parameters is as follows:

1. `a` is the first operand for the reduction operation.
2. `b` is the second operand for the reduction operation.

#### Lambda function keep

The lambda function `keep` is new one compared to the flexible reduction for arrays, vectors or other linear structures. The reason is that the result consists of as many numbers as there are matrix rows. Result obtained for each matrix row is processed by this lambda function. It is declared as follows:

\includelineno snippet_rows_reduction_keep_declaration.cpp

The meaning of the particular parameters is as follows:

1. `rowIdx` is an index of the matrix row related to given result of flexible reduction.
2. `value`is the result of the flexible reduction in given matrix row.

The method `reduceRows` (\ref TNL::Matrices::DenseMatrix::reduceRows, \ref TNL::Matrices::SparseMatrix::reduceRows, \ref TNL::Matrices::TridiagonalMatrix::reduceRows, \ref TNL::Matrices::MultidiagonalMatrix::reduceRows, \ref TNL::Matrices::LambdaMatrix::reduceRows) accepts the following arguments:

1. `begin` is the beginning of the matrix rows range on which the reduction will be performed.
2. `end` is the end of the matrix rows range on which the reduction will be performed. The last matrix row which is going to be processed has index `end-1`.
3. `fetch` is the lambda function for data fetching.
4. `reduce` is the the lambda function performing the reduction.
5. `keep` is the lambda function responsible for processing the results from particular matrix rows.
6. `zero` is the "zero" element of given reduction operation also known as *idempotent*.

Though the interface is the same for all matrix types, in the following part we will show several examples for different matrix types to better demonstrate possible ways of use of the flexible reduction for matrices.

### Dense matrices example

The following example demonstrates implementation of the dense matrix-vector product \f$ {\bf y} = A \vec {\bf x}\f$, i.e.

\f[
   y_i = \sum_{j=0}^{columns - 1} a_{ij} x_j \text{ for } i = 0, \ldots, rows-1.
\f]

\includelineno DenseMatrixExample_reduceRows_vectorProduct.cpp

We set the following lambda functions:

* `fetch` lambda function computes the product \f$ a_{ij}x_j\f$ where \f$ a_{ij} \f$ is represented by `value` and \f$x_j \f$ is represented by `xView[columnIdx]` (line 40).
* `reduce` - reduction is just sum of particular products and it is represented by \ref std::plus (line 53).
* `keep` is responsible for storing the results of reduction in each matrix row (which is represented by the variable `value`) into the output vector `y`.

The result looks as:

\include DenseMatrixExample_reduceRows_vectorProduct.out

We will show one more example which is a computation of maximal absolute value in each matrix row. The results will be stored in a vector:

\f[
y_i = \max_{j=1,\ldots,n} |a_{ij}|.
\f]

See the following example:

\includelineno DenseMatrixExample_reduceRows_maxNorm.cpp

The lambda functions rare:

* `fetch` lambda function just returns absolute value of \f$a_{ij} \f$ which is represented again by the variable `value`.
* `reduce` lambda function returns larger of given values.
* `keep` stores the results to the output vector the same way as in the previous example.

Note, that the idempotent value for the reduction is \ref std::numeric_limits< double >::lowest. Of course, if we compute the maximum of all output vector elements, we get some kind of maximal matrix norm. The output looks as:

\include DenseMatrixExample_reduceRows_maxNorm.out

### Sparse matrices example

The following example demonstrates sparse matrix-vector product:

\includelineno SparseMatrixExample_reduceRows_vectorProduct.cpp

On the lines 11-16 we set the following matrix:

\f[
\left(
\begin{array}{ccccc}
1 & . & . & . & . \\
1 & 2 & . & . & . \\
. & 1 & 8 & . & . \\
. & . & 1 & 9 & . \\
. & . & . & . & 1
\end{array}
\right)
\f]

The lambda functions on the lines 39-48 are the same as in the example with the dense matrix. The result looks as follows:

\include SparseMatrixExample_reduceRows_vectorProduct.out

### Tridiagonal matrices example

In this example, we will compute maximal absolute value in each row of the following tridiagonal matrix:

\f[
\left(
\begin{array}{ccccc}
1 & 3 &   &   &   &    \\
2 & 1 & 3 &   &   &    \\
  & 2 & 1 & 3 &   &    \\
  &   & 2 & 1 & 3 &    \\
  &   &   & 2 & 1 & 3
\end{array}
\right).
\f]

The source code reads as follows:

\includelineno TridiagonalMatrixExample_reduceRows.cpp

Here we first set the tridiagonal matrix (lines 10-27). Next we allocate the vector `rowMax` where we will store the results (line 32). The lambda function are:

* `fetch` (lines 42-44) is responsible for reading the matrix elements. In our example, the only thing this function has to do, is to compute the absolute value of each matrix element represented by variable `value`.
* `reduce` (lines 49-51), performs reduction operation. In this case, it returns maximum of two input values `a` and `b`.
* `keep` (lines 56-58) takes the result of the reduction in variable `value` in each row and stores it into the vector `rowMax` via related vector view `rowMaxView`.

Note, that the idempotent value for the reduction is \ref std::numeric_limits< double >::lowest. The results looks as follows:

\include TridiagonalMatrixExample_reduceRows.out

### Multidiagonal matrices example

The next example computes again the maximal absolute value in each row. Now, we do it for multidiagonal matrix the following form:

\f[
\left(
\begin{array}{ccccc}
1  &   &   &   &  \\
2  & 1 &   &   &  \\
3  & 2 & 1 &   &  \\
   & 3 & 2 & 1 &  \\
   &   & 3 & 2 & 1
\end{array}
\right)
\f]

We first create vector `rowMax` into which we will store the results and fetch it view `rowMaxView` (line 39). Next we prepare necessary lambda functions:

* `fetch` (lines 44-46) is responsible for reading the matrix element value which is stored in the constant reference `value` and for returning its absolute value. The other parameters `rowIdx` and `columnIdx` correspond to row and column indexes respectively and they are omitted in our example.
* `reduce` (lines 51-53) returns maximum value of the two input values `a` and `b`.
* `keep` (line 58-60) stores the input `value` at the corresponding position, given by the row index `rowIdx`, in the output vector view `rowMaxView`.

Finally, we call the method `reduceRows` (\ref TNL::Matrices::MultidiagonalMatrix::reduceRows) with parameters telling the interval of rows to be processed (the first and second parameter), the lambda functions `fetch`, `reduce` and `keep`, and the idempotent element for the reduction operation which is the lowest number of given type (\ref std::numeric_limits< double >::lowest ). The result looks as follows:

\include MultidiagonalMatrixExample_reduceRows.out

### Lambda matrices example

The reduction of matrix rows is available for the lambda matrices as well. See the following example:

\includelineno LambdaMatrixExample_reduceRows.cpp

On the lines 14-21, we create the lower triangular lambda matrix which looks as follows:

\f[
\left(
\begin{array}{ccccc}
1 &   &   &   &   \\
2 & 1 &   &   &   \\
3 & 2 & 1 &   &   \\
4 & 3 & 2 & 1 &   \\
5 & 4 & 3 & 2 & 1
\end{array}
\right)
\f]

We want to compute maximal absolute value of matrix elements in each row. For this purpose we define well known lambda functions:

* `fetch` takes the value of the lambda matrix element and returns its absolute value.
* `reduce` computes maximum value of two input variables.
* `keep` stores the results into output vector `rowMax`.

Note that the interface of the lambda functions is the same as for other matrix types. The result looks as follows:

\include LambdaMatrixExample_reduceRows.out

## Matrix-vector product

One of the most important matrix operation is the matrix-vector multiplication. It is represented by a method `vectorProduct` (\ref TNL::Matrices::DenseMatrix::vectorProduct, \ref TNL::Matrices::SparseMatrix::vectorProduct, \ref TNL::Matrices::TridiagonalMatrix::vectorProduct, \ref TNL::Matrices::MultidiagonalMatrix::vectorProduct, \ref TNL::Matrices::LambdaMatrix::vectorProduct). It is templated method with two template parameters `InVector` and `OutVector` telling the types of input and output vector respectively. Usually one will substitute some of \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or \ref TNL::Containers::VectorView for these types. The method accepts the following parameters:

1. `inVector` is the input vector having the same number of elements as the number of matrix columns.
2. `outVector` is the output vector having the same number of elements as the number of matrix rows.
3. `matrixMultiplicator` is a number by which the result of matrix-vector product is multiplied.
4. `outVectorMultiplicator` is a number by which the output vector is multiplied before added to the result of matrix-vector product.
5. `begin` is the beginning of the matrix rows range on which we compute the matrix-vector product.
6. `end` is the end of the matrix rows range on which the matrix-vector product will be evaluated. The last matrix row which is going to be processed has index `end-1`.

Note that the output vector dimension must be the same as the number of matrix rows no matter how we set `begin` and `end` parameters. These parameters just say that some matrix rows and the output vector elements are omitted.

To summarize, this method computes the following formula:

`outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector.`

## Matrix I/O operations

All  matrices can be saved to a file using a method `save` (\ref TNL::Matrices::DenseMatrix::save, \ref TNL::Matrices::SparseMatrix::save, \ref TNL::Matrices::TridiagonalMatrix::save, \ref TNL::Matrices::MultidiagonalMatrix::save, \ref TNL::Matrices::LambdaMatrix::save) and restored with a method `load` (\ref TNL::Matrices::DenseMatrix::load, \ref TNL::Matrices::SparseMatrix::load, \ref TNL::Matrices::TridiagonalMatrix::load, \ref TNL::Matrices::MultidiagonalMatrix::load, \ref TNL::Matrices::LambdaMatrix::load). To print the matrix, there is a method `print` (\ref TNL::Matrices::DenseMatrix::print, \ref TNL::Matrices::SparseMatrix::print, \ref TNL::Matrices::TridiagonalMatrix::print, \ref TNL::Matrices::MultidiagonalMatrix::print, \ref TNL::Matrices::LambdaMatrix::print) can be used.

### Matrix reader and writer

TNL also offers matrix reader (\ref TNL::Matrices::MatrixReader) and matrix writer (\ref TNL::Matrices::MatrixWriter) for import and export of matrices respectively. The matrix reader currently supports only [Coordinate MTX file format](https://math.nist.gov/MatrixMarket/formats.html#coord) which is popular mainly for sparse matrices. By the mean of the matrix writer, we can export TNL matrices into coordinate MTX format as well. In addition, the matrices can be exported to a text file suitable for [Gnuplot program](http://www.gnuplot.info/) which can be used for matrix visualization. Finally, a pattern of nonzero matrix elements can be visualized via the EPS format - [Encapsulated PostScript](https://en.wikipedia.org/wiki/Encapsulated_PostScript). We demonstrate both matrix reader and writer in the following example:

\includelineno MatrixWriterReaderExample.cpp

The example consists of two functions - `matrixWriterExample` (lines 10-24) and `matrixReaderExample` (lines 36-54). In the first one, we first create a toy matrix (lines 13-22) which we subsequently export into Gnuplot (line 26), EPS (line 29) and MTX (line 32) formats. In the next step (the `matrixReaderExample` function on lines 36-54), the MTX file is used to import the matrix into sparse (line 43) and dense (line 51) matrices. Both matrices are printed out (lines 45 and 53).

The result looks as follows:

\includelineno MatrixWriterReaderExample.out

## Appendix

### Benchmark of dense matrix setup

\includelineno DenseMatrixSetup_Benchmark.cpp

### Benchmark of sparse matrix setup

\includelineno SparseMatrixSetup_Benchmark.cpp

### Benchmark of multidiagonal matrix setup

\includelineno MultidiagonalMatrixSetup_Benchmark.cpp
