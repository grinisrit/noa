# Segments tutorial

[TOC]


## Introduction

*Segments* represent data structure for manipulation with several local arrays (denoted also as segments) having different size in general. All the local arrays are supposed to be allocated in one continuos global array. The data structure segments offers mapping between indexes of particular local arrays and indexes of the global array. Segments do not store any data, segments just represent a layer for efficient access and operations with group of segments of linear containers (i.e. local arrays) with different size in general. One can perform parallel operations like *for* or *flexible reduction* on particular segments (local arrays).

A typical example of *segments* are different formats for sparse matrices. Sparse matrix like the following
 \f[
  \left(
  \begin{array}{ccccc}
   1  &  0  &  2  &  0  &  0 \\
    0  &  0  &  5  &  0  &  0 \\
    3  &  4  &  7  &  9  &  0 \\
    0  &  0  &  0  &  0  & 12 \\
   0  &  0  & 15  & 17  & 20
  \end{array}
  \right)
 \f]
 is usually first compressed which means that the zero elements are omitted to get the following "matrix":

 \f[
 \begin{array}{ccccc}
    1  &   2  \\
    5   \\
    3  &   4  &  7 &  9   \\
    12 \\
    15 & 17  & 20
 \end{array}
 \f]
 We have to store column index of each matrix elements as well in a "matrix" like this:
 \f[
 \begin{array}{ccccc}
    0  &   2  \\
    2   \\
    0  &   1  &  2 &  3   \\
    4 \\
    2 & 3  & 4
 \end{array}
 \f]

 Such "matrices" can be stored in memory in a row-wise manner in one contiguous array because of the performance reasons. The first "matrix" (i.e. values of the matrix elements)  would be stored as follows

 \f[
    \begin{array}{|cc|c|cccc|c|cc|} 1 & 2 &  5 & 3 & 4 & 7 & 9 & 12 & 15 & 17 & 20 \end{array}
 \f]

and the second one (i.e. column indexes of the matrix values) as follows

\f[
    \begin{array}{|cc|c|cccc|c|cc|} 0 & 2 & 2 & 0 & 1 & 2 & 3 & 4 & 2 & 3 & 4 \end{array}
 \f]

What we see above is so called [CSR sparse matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)). It is the most popular format for storage of sparse matrices designed for high performance. However, it may not be the most efficient format for storage of sparse matrices on GPUs. Therefore many other formats have been developed to get better performance. These formats often have different layout of the matrix elements in the memory. They have to deal especially with two difficulties:

1. Efficient storage of matrix elements in the memory to fulfill the requirements of coalesced memory accesses on GPUs or good spatial locality for efficient use of caches on CPUs.
2. Efficient mapping of GPU threads to different matrix rows.

TNL offers the following sparse matrix formats in a form of segments (Ellpack formats often use so called *padding elements* like padding zeros in terms of sparse matrices):

1. [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) (\ref TNL::Algorithms::Segments::CSR) is the most popular format for sparse matrices. It is simple ane very efficient especially on CPUs and today there are efficient kernels even for GPUs. The following GPU kernels are implemented in TNL:
   1. [Scalar](http://mgarland.org/files/papers/nvr-2008-004.pdf) which maps one GPU thread for each segment (matrix row).
   2. [Vector](http://mgarland.org/files/papers/nvr-2008-004.pdf) which maps one warp of GPU threads for each segment (matrix row).
   3. [Adaptive](https://ieeexplore.ieee.org/document/7397620) ...
2. [Ellpack format](http://mgarland.org/files/papers/nvr-2008-004.pdf) (\ref TNL::Algorithms::Segments::Ellpack) uses padding elements to have the same number of element in each segment. It can be highly inefficient in cases when one works with few very long segments.
3. [SlicedEllpack format](https://link.springer.com/chapter/10.1007/978-3-642-11515-8_10) (\ref TNL::Algorithms::Segments::SlicedEllpack) which was also presented as [Row-grouped CSR format](https://arxiv.org/abs/1012.2270) is similar to common Ellpack. However, SlicedEllpack first merges the segments into groups of 32. It also uses padding elements but only segments within the same group are aligned to have the same size. Therefore there is not such a high performance drop because of few long segments.
4. [ChunkedEllpack format](http://geraldine.fjfi.cvut.cz/~oberhuber/data/vyzkum/publikace/12-heller-oberhuber-improved-rgcsr-format.pdf) (\ref TNL::Algorithms::Segments::ChunkedEllpack) is simillar to SlicedEllpack but it splits segments into chunks which allows to map more GPU threads to one segment.
5. [BiEllpack format](https://www.sciencedirect.com/science/article/pii/S0743731514000458?casa_token=2phrEj0Ef1gAAAAA:Lgf6rMBUN6T7TJne6mAgI_CSUJ-jR8jz7Eghdv6L0SJeGm4jfso-x6Wh8zgERk3Si7nFtTAJngg) (\ref TNL::Algorithms::Segments::BiEllpack) is simillar to ChunkedEllpack. In addition it sorts segments within the same slice w.r.t. their length to achieve higher performance and better memory accesses.

Especially in case of GPUs, the performance of each format strongly depends on distribution of the segment sizes. Therefore we cannot say that one of the previous formats would outperform the others in general. To get the best performance, one should try more of the formats and choose the best one. It is the reason why TNL offers more of them and additional formats will acrue.

Necessity of working with this kind of data structures is not limited only to sparse matrices. We could name at least few other applications for segments:

1. [Graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) - one segment represents one graph node, the elements in one segments are indexes of its neighbors.
2. [Unstructured numerical meshes](https://en.wikipedia.org/wiki/Types_of_mesh) - unstructured numerical mesh is a graph in fact.
3. [Particle in cell method](https://en.wikipedia.org/wiki/Particle-in-cell) - one segment represents one cell, the elements in one segment are indexes of the particles.
4. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) - segments represent one cluster, the elements represent vectors belonging to given cluster.
5. [Hashing](https://arxiv.org/abs/1907.02900) - segments are particular rows of the hash table, elements in segments corresponds with colliding hashed elements.

In general, segments can be used for problems that somehow corresponds wit 2D data structure where each row can have different size and we need to perform miscellaneous operations within the rows. The name *segments* comes from segmented parallel reduction or [segmented scan (prefix-sum)](https://en.wikipedia.org/wiki/Segmented_scan).

## Segments setup

Segments are defined just by sizes of particular segments. The following example shows how to create them:

\includelineno Algorithms/Segments/SegmentsPrintingExample-1.cpp

We use constructor with initializer list (line 16) where each element of the list defines size of one segment. Next we print sizes of particular segments (line 17). We call this function for different segments types (excluding \ref TNL::Algorithms::Segments::SlicedEllpack since it would behave the same way as \ref TNL::Algorithms::Segments::Ellpack on this small example). The result looks as follows:

\include SegmentsPrintingExample-1.out

We can see, that real sizes of the segments are different for all Ellpack-based formats. As we said already, these formats often use padding elements to get more efficient access to the memory. For example \ref TNL::Algorithms::Segments::ChunkedEllpack format involves multiple of elements. It is, however, only because of very small example we present now, on large examples the overhead is not so significant.

We remind that segments represent rather sparse format then data structure because they do not store any data. The following example shows how to connect segments with array:

\includelineno Algorithms/Segments/SegmentsPrintingExample-2.cpp

On the line 19, we show how to create segments with vector (\ref TNL::Containers::Vector) carrying the segments sizes. Of course, the same constructor works even for arrays and views (i.e. \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView and \ref TNL::Containers::VectorView). Next we print the real segment sizes depending on the format in the background (line 20) the same way as we did in the previous example. On the line 25, we allocate array having the size requested by the `segments` by means of method `getStorageSize` (\ref TNL::Algortihms::Segments::CSR::getStorageSize for example). This method says how many elements the segments need to be able to address all elements by their global index. On the lines 26-28, we mark each element of the array by its rank in the array. On the line 35, we use function \ref TNL::Algorithms::Segments::printSegments which accepts lambda function `fetch` as one its parameters. The lambda function reads data from our array `data` (with the help of array view `data_view`) according to given global index `globalIdx` (line 34). The result looks as follows:

\include SegmentsPrintingExample-2.out

Frankly, what we see is not so important. It only shows that different segments formats can use very different mapping of elements identified by its *segment index* and *local index* (rank of the element in given segment) to a *global index* which serves as an address in the related container.

## Iteration over elements of segments

In this section, we show how to iterate over the elements of segments and how to manipulate with them. There are three possible ways:

1. Method `forElements` (\ref TNL::Algorihms::Segments::CSR::forElements for example), which iterates in parallel over all elements of segments and perform given lambda function on each of them.
2. Method `forSegments` (\ref TNL::Algorihms::Segments::CSR::forSegments for example), which iterates in parallel over all segments. It is better choice when we need to process each segment sequentially are we have significant amount of computations common for all elements in each segment.
3. Method `sequentailForSegments` (\ref TNL::Algorihms::Segments::CSR::sequentailForSegments for example), which iterates over all segments sequentially i.e. using only one thread even on GPUs. It is useful for debugging or for printing for example.

Methods iterating over particular segments use a segment view (\ref TNL::Algorithms::Segments::SegmentView) to access the elements of given segment. The segment view offers iterator for better convenience.

### Method forElements

The following example shows use of the method `forElements`:

\includelineno Algorithms/Segments/SegmentsExample_forElements.cpp

On the line 7, we first create segments with linearly increasing size (so it is like lower triangular matrix). Next, we allocate array `data` (line 21) having the same size as the number of elements managed by the segments. It can be obtained by the method `getStorageSize` (\ref TNL::Algorithms::Segments::CSR::getStorageSize for example). We prepare array view `data_view` for the purpose of use in lambda functions (line 26). Finally, we call the method `forAllElements` (lines 27-29) which iterates in parallel over all elements in the segments and for each element it calls given lambda function. The lambda function receives three arguments - `segmentIdx` is an index of the segment the element belongs to, `localIdx` is the rank of the element within the segment and `globalIdx` is an index of the element in the array `data`. We use the global index to set proper element of the array `data` to the index of the segment. On the line 35, we print the array `data`. We can see elements belonging to particular segments by their indexes. The layout of the elements depends on the type of segments (which means sparse format in use). Next we print the elements of array `data` by segments (lines 36 and 37). The function `printSegments` iterates over all elements and it reads the elements of the array `data` with the help of the lambda function defined on the line 36.

Note, that for the Ellpack format, the output looks as follows:

```
Seg. 0: [ 0, 0, 0, 0, 0 ]
Seg. 1: [ 1, 1, 1, 1, 1 ]
Seg. 2: [ 2, 2, 2, 2, 2 ]
Seg. 3: [ 3, 3, 3, 3, 3 ]
Seg. 4: [ 4, 4, 4, 4, 4 ]
```

We see more elements that we have requested. The reason is that the Ellpack format uses padding elements for optimizing access to the memory. Segments give access even to the padding elements, they can be used in case when we get to situation of need of additional elements. Therefore we need to check for relevant and padding elements each time we work with elements of segments. It is demonstrated on the lines 43-46 where we set the array `data` again but we check for the padding elements (line 44). After printing the segments the same way as before (line 53) we get correct result:

```
Seg. 0: [ 0, 0, 0, 0, 0 ]
Seg. 1: [ 1, 1, 0, 0, 0 ]
Seg. 2: [ 2, 2, 2, 0, 0 ]
Seg. 3: [ 3, 3, 3, 3, 0 ]
Seg. 4: [ 4, 4, 4, 4, 4 ]
```

The result of the whole example looks as follows:

\include SegmentsExample_forElements.out

### Method forSegments

Method `forSegments` iterates in parallel over particular segments. Iteration over elements within the segment is sequential. There are two reasons for such proceeding:

1. The iteration over the elements within the same segments must be sequential, i.e. the computation with one element depends on a result of the computation with the previous one.
2. Some part of computations on all elements in one segment is common. In this case, we can first perform the common part and then iterate over the elements. If we would use the method `forElements`, the common part would have to be performed for each element.

#### Sequential dependency

The first situation is demonstrated in the following example:

\includelineno Algorithms/Segments/SegmentsExample_forSegments-1.cpp

The result looks as follows:

The code is the same as in the previous example up to line 26. Instead of calling the method `forElements` we call the method `forSegments` (line 28) for which we need to define type  `SegmentViewType` (\ref TNL::Algorithms::Segments::CSR::SegmentViewType for example). The lambda function on the line 28 gets the segment view and it iterates over all elements of the segment by means of a for loop. We use auxiliary variable `sum` to compute cumulative sum of elements in each segment which is just the sequential dependency. The result looks as follows:

\include SegmentsExample_forSegments-1.out

#### Common computations

Now let's take a look at the second situation, i.e. there are common computations for all elements of one segment. In the following example, we first set values of each element using the method `forElements` which we are already familiar with (lines 26-29). Next we print values of all elements (lines 34-36) and then we use the method `forAllSegments` (lines 41-52) to divide each element by a sum of values of all elements in a segment. So we first sum up all elements in the segment (lines 43-47). This is the common part of the computation for all elements in the segment. Next we perform the division of all elements by the value of the variable `sum` (lines 48-51).

\includelineno Algorithms/Segments/SegmentsExample_forSegments-2.cpp

The result looks as follows:

\include SegmentsExample_forSegments-2.out

## Flexible reduction within segments

In this section we will explain extension of [flexible reduction]() to segments. It allows to reduce all elements within the same segment and store the result into an array. See the following example:

\includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp

We first create the segments `segments` (line 18), related array `data` (line 23) and setup the elements (lines 28-32). After printing the segments (lines 37-39) we are ready for the parallel reduction. It requires three lambda fuctions:

1. `fetch` which reads data belonging to particular elements of the segments. The fetch function can have two different forms - *brief* and *full*:
   * *Brief form* - is this case the lambda function gets only global index and the `compute` flag:
```
      auto fetch = [=] __cuda_callable__ ( int globalIdx, bool& compute ) -> double { ... };
```
   * *Full form* - in this case the lambda function receives even the segment index and element index:
```
      auto fetch = [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx, bool& compute ) -> double { ... };
```
   where `segmentIdx` is the index of the segment, `localIdx` is the rank of the element within the segment, `globalIdx` is index of the element in the related array and `compute` serves for the reduction interruption which means that the remaining elements in the segment can be omitted. Many formats used for segments are optimized for much higher performance if the brief variant is used. The form of the `fetch` lambda function is detected automatically using [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) and so the use of both is very ease for the user.
2. `reduce` is a function representing the reduction operation, in our case it is defined as follows:
```
auto reduce = [=] __cuda_callable__ ( const double& a, const double& b ) -> double { return a + b; }
```
   or, in fact, we can use the function `std::plus`.
3. `keep` is a lambda function responsible for storage of the results. It is supposed to be defined as:
```
auto keep = [=] __cuda_callable__ ( int segmentIdx, const double& value ) mutable { ... };
```
where `segmentIdx` is an index of the segment of which the reduction result we aim to store and `value` is the result of the reduction in the segment.

We first create vector `sums` where we will store the results (line 44) and prepare a view to this vector for later use in the lambda functions. We demonstrate use of both variants - full by `fetch_full` (lines 46-54) and brief by `fetch_brief` (lines 55-57). The lambda function `keep` for storing the sums from particular segments into the vector `sums` is on the lines 59-60. Finally, we call the method `reduceAllSegments` (\ref TNL::Algorithms::Segments::CSR::reduceSegments for example) to compute the reductions in the segments - first with  `fetch_full` (line 61) and then with `fetch_brief` (line 63). In both cases, we use `std::plus` for the reduction and we pass zero (the last argument) as an idempotent element for sumation. In both cases we print the results which are supposed to be the same. The result looks as follows:

\include SegmentsExample_reduceSegments.out




