// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
   namespace Algorithms {
/**
 * \brief Namespace holding segments data structures.
 *
 * *Segments* represent data structure for manipulation with several local arrays (denoted also as segments)
 having different size in general. All the local arrays are supposed to be allocated in one continuos global array.
 The data structure segments offers mapping between indexes of particular local arrays and indexes
 of the global array. In addition,one can perform parallel operations like for or flexible reduction on partical
 local arrays.

 A typical example for use of *segments* is implementation of sparse matrices. Sparse matrix like the following
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

 Such "matrices" can be stored in memory in a row-wise manner in one contiguous array because of the performance reasons. The first "matrix" (i.e. values of the matrix elements)
 would be stored as follows

 \f[
    \begin{array}{|cc|c|cccc|c|cc|} 1 & 2 &  5 & 3 & 4 & 7 & 9 & 12 & 15 & 17 & 20 \end{array}
 \f]

and the second one (i.e. column indexes of the matrix values) as follows

\f[
    \begin{array}{|cc|c|cccc|c|cc|} 0 & 2 & 2 & 0 & 1 & 2 & 3 & 4 & 2 & 3 & 4 \end{array}
 \f]

What we see above is so called [CSR sparse matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).
It is the most popular format for storage of sparse matrices designed for high performance. However, it may not be the most efficient format for storage
of sparse matrices on GPUs. Therefore many other formats have been developed to get better performance. These formats often have different layout
of the matrix elements in the memory. They have to deal especially with two difficulties:

1. Efficient storage of matrix elements in the memory to fulfill the requirements of coalesced memory accesses on GPUs or good spatial locality
 for efficient use of caches on CPUs.
2. Efficient mapping of GPU threads to different matrix rows.

Necessity of working with this kind of data structure is not limited only to sparse matrices. We could name at least few others:

1. Efficient storage of [graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) - one segment represents one graph node,
   the elements in one segments are indexes of its neighbors.
2. [Unstructured numerical meshes](https://en.wikipedia.org/wiki/Types_of_mesh) - unstructured numerical mesh is a graph in fact.
3. [Particle in cell method](https://en.wikipedia.org/wiki/Particle-in-cell) - one segment represents one cell, the elements in one segment
   are indexes of the particles.
4. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) - segments represent one cluster, the elements represent vectors
   belonging to given cluster.
5. [Hashing](https://arxiv.org/abs/1907.02900) - segments are particular rows of the hash table, elements in segments corresponds with coliding
   hashed elements.

In general, segments can be used for problems that somehow corresponds wit 2D data structure where each row can have different size and we need
to perform miscellaneous operations within the rows. The name *segments* comes from segmented parallel reduction or
[segmented scan (prefix-sum)](https://en.wikipedia.org/wiki/Segmented_scan).

The following example demonstrates the essence of *segments* in TNL:

\includelineno Algorithms/Segments/SegmentsExample_General.cpp

We demonstrate two formats of segments - \ref noa::TNL::Algorithms::Segments::CSR and \ref noa::TNL::Algorithms::Segments::Ellpack running on both CPU and GPU
(lines 58-76). For each of them, we call function `SegmentsExample` which first creates given segments (line 18). The segments are defined by the sizes of
particular segments.

Next we allocate array with data related to the segments (line 24). The number of elemets managed by the segments is given by
\ref noa::TNL::Algorithms::Segments::CSR::getStorageSize and \ref noa::TNL::Algorithms::Segments::Ellpack::getStorageSize respectively.

Next we setup the segments elements (lines 29-33) by calling \ref noa::TNL::Algorithms::Segments::CSR::forAllElements
(and \ref noa::TNL::Algorithms::Segments::CSR::forAllElements respectively) which iterates over all elements of the segments
in parallel and perform given lambda function. The lambda function receives index of the segment (`segmentIdx`),
index of the element within the segment (`localIdx`), index of the element within the array `data` and a reference to boolean (`compute`) which serves as a
hint for interrupting the iteration over the elements of given segment when it is set to `false`. The value of the elements having the local index smaller or equal
to the segments index is set to the value of the segment index. It creates, in fact, lower triangular matrix elements of which have values equal to row index.

Next we use a function \ref noa::TNL::Algorithms::Segments::printSegments to print the content of the segments (lines 38-39). To do this we have to provide a lambda function
`fetch` (line 38) which returns value of elements with given global index.

Finally we show how to compute sum of all elemnts in each segment. Firstly, we create vector into which we will store the sums (line 44) and get its view (line 45).
The size of the vector is given by the number of the segments which can be obtained by the means of the method \ref noa::TNL::Algorithms::Segments::CSR::getSegmentsCount
(and \ref noa::TNL::Algorithms::Segments::Ellpack::getSegmentsCount respectively). The sums are computed using the method \ref noa::TNL::Algorithms::Segments::CSR::reduceAllSegments
(and \ref noa::TNL::Algorithms::Segments::Ellpack::reduceAllSegments respectively) which works the same way as the flexible parallel reduction (\ref noa::TNL::Algorithms::Reduction).
It requires lambda functions `fetch` for reading the data related to particular elements of the segments, function `reduce` which is \ref std::plus in this case and a
function `keep` to store the result of sums in particular segments.

The result looks as follows:

\include SegmentsExample_General.out

Note that the Ellpack format manages more elements than we asked for. It is because some formats use padding elements for more efficient memory accesses. The padding
elements are available to the user as well and so we must ensure that work only with those elements we want to. This is the reason why we use the if statement on the
line 31 when setting up the values of the elements in segments. The padding elements can be used in case when we later need more elements than we requested. However,
the segments data structure does not allow any resizing of the segments. One can change the sizes of the segments, however, the access to the originally managed data
is becoming invalid at that moment.

*/



      namespace Segments {

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
