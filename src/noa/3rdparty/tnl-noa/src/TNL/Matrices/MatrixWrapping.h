// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSR.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Ellpack.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrixView.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Function for wrapping an array of values into a dense matrix view.
 *
 * \tparam Device is a device on which the array is allocated.
 * \tparam Real is a type of array elements.
 * \tparam Index is a type for indexing of matrix elements.
 * \tparam Organization is matrix elements organization - see \ref noa::TNL::Algorithms::Segments::ElementsOrganization.
 * \param rows is a number of matrix rows.
 * \param columns is a number of matrix columns.
 * \param values is the array with matrix elements values.
 * \return instance of DenseMatrixView wrapping the array.
 *
 * The array size must be equal to product of `rows` and `columns`. The dense matrix view does not deallocate the input
 * array at the end of its lifespan.
 *
 * \par Example
 * \include Matrices/DenseMatrix/DenseMatrixViewExample_wrap.cpp
 * \par Output
 * \include DenseMatrixViewExample_wrap.out
 */
template< typename Device,
          typename Real,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
DenseMatrixView< Real, Device, Index, Organization >
wrapDenseMatrix( const Index& rows, const Index& columns, Real* values )
{
   using MatrixView = DenseMatrixView< Real, Device, Index, Organization >;
   using ValuesViewType = typename MatrixView::ValuesViewType;
   return MatrixView( rows, columns, ValuesViewType( values, rows * columns ) );
}

/**
 * \brief Function for wrapping of arrays defining [CSR
 * format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) into a sparse matrix
 * view.
 *
 * \tparam Device  is a device on which the arrays are allocated.
 * \tparam Real is a type of matrix elements values.
 * \tparam Index is a type for matrix elements indexing.
 * \param rows is a number of matrix rows.
 * \param columns is a number of matrix columns.
 * \param rowPointers is an array holding row pointers of the CSR format ( `ROW_INDEX`
 * [here](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))) \param values is an
 * array with values of matrix elements ( `V`
 * [here](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))) \param columnIndexes is
 * an array with column indexes of matrix elements  ( `COL_INDEX`
 * [here](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))) \return instance of
 * SparseMatrixView with CSR format.
 *
 * The size of array \e rowPointers must be equal to number of `rows + 1`. The last element of the array equals to the number of
 * all nonzero matrix elements. The sizes of arrays `values` and `columnIndexes` must be equal to this number.
 *
 * \par Example
 * \include Matrices/SparseMatrix/SparseMatrixViewExample_wrapCSR.cpp
 * \par Output
 * \include SparseMatrixViewExample_wrapCSR.out
 */
template< typename Device, typename Real, typename Index >
SparseMatrixView< Real, Device, Index, GeneralMatrix, Algorithms::Segments::CSRViewDefault >
wrapCSRMatrix( const Index& rows, const Index& columns, Index* rowPointers, Real* values, Index* columnIndexes )
{
   using MatrixView = SparseMatrixView< Real, Device, Index, GeneralMatrix, Algorithms::Segments::CSRViewDefault >;
   using ValuesViewType = typename MatrixView::ValuesViewType;
   using ColumnIndexesView = typename MatrixView::ColumnsIndexesViewType;
   using SegmentsView = typename MatrixView::SegmentsViewType;
   using KernelView = typename SegmentsView::KernelView;
   using RowPointersView = typename SegmentsView::OffsetsView;
   RowPointersView rowPointersView( rowPointers, rows + 1 );
   Index elementsCount = rowPointersView.getElement( rows );
   SegmentsView segments( rowPointersView, KernelView() );
   ValuesViewType valuesView( values, elementsCount );
   ColumnIndexesView columnIndexesView( columnIndexes, elementsCount );
   return MatrixView( rows, columns, valuesView, columnIndexesView, segments );
}

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Device, ElementsOrganization Organization, typename Real, typename Index, int Alignment = 1 >
struct EllpackMatrixWrapper
{
   template< typename Device_, typename Index_ >
   using EllpackSegments = Algorithms::Segments::EllpackView< Device_, Index_, Organization, Alignment >;
   using MatrixView = SparseMatrixView< Real, Device, Index, GeneralMatrix, EllpackSegments >;

   static MatrixView
   wrap( const Index& rows, const Index& columns, const Index& nonzerosPerRow, Real* values, Index* columnIndexes )
   {
      using ValuesViewType = typename MatrixView::ValuesViewType;
      using ColumnIndexesView = typename MatrixView::ColumnsIndexesViewType;
      using SegmentsView = Algorithms::Segments::EllpackView< Device, Index, Organization, Alignment >;
      SegmentsView segments( rows, nonzerosPerRow );
      Index elementsCount = segments.getStorageSize();
      ValuesViewType valuesView( values, elementsCount );
      ColumnIndexesView columnIndexesView( columnIndexes, elementsCount );
      return MatrixView( rows, columns, valuesView, columnIndexesView, segments );
   }
};
/// \endcond

/**
 * \brief Function for wrapping of arrays defining [Ellpack
 * format](https://people.math.sc.edu/Burkardt/data/sparse_ellpack/sparse_ellpack.html) into a sparse matrix view.
 *
 * \tparam Device  is a device on which the arrays are allocated.
 * \tparam Real is a type of matrix elements values.
 * \tparam Index is a type for matrix elements indexing.
 * \tparam Alignment defines alignment of data. The number of matrix rows is rounded to a multiple of this number. It it usefull
 * mainly for GPUs. \param rows is a number of matrix rows. \param columns is a number of matrix columns. \param nonzerosPerRow
 * is number of nonzero matrix elements in each row. \param values is an array with values of matrix elements. \param
 * columnIndexes is an array with column indexes of matrix elements. \return instance of SparseMatrixView with CSR format.
 *
 *  The sizes of arrays `values` and `columnIndexes` must be equal to `rows * nonzerosPerRow`. Use `-1` as a column index for
 * padding zeros.
 *
 * \par Example
 * \include Matrices/SparseMatrix/SparseMatrixViewExample_wrapEllpack.cpp
 * \par Output
 * \include SparseMatrixViewExample_wrapEllpack.out
 */
template< typename Device, ElementsOrganization Organization, typename Real, typename Index, int Alignment = 1 >
auto
wrapEllpackMatrix( const Index rows, const Index columns, const Index nonzerosPerRow, Real* values, Index* columnIndexes )
   -> decltype( EllpackMatrixWrapper< Device, Organization, Real, Index, Alignment >::wrap( rows,
                                                                                            columns,
                                                                                            nonzerosPerRow,
                                                                                            values,
                                                                                            columnIndexes ) )
{
   return EllpackMatrixWrapper< Device, Organization, Real, Index, Alignment >::wrap(
      rows, columns, nonzerosPerRow, values, columnIndexes );
}

}  // namespace Matrices
}  // namespace noa::TNL
