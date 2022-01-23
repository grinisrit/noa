// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/TNL/Matrices/MatrixRowViewIterator.h>
#include <noa/3rdparty/TNL/Matrices/details/SparseMatrixRowViewValueGetter.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of sparse matrix.
 *
 * \tparam SegmentView is a segment view of segments representing the matrix format.
 * \tparam ValuesView is a vector view storing the matrix elements values.
 * \tparam ColumnsIndexesView is a vector view storing the column indexes of the matrix element.
 *
 * See \ref SparseMatrix and \ref SparseMatrixView.
 *
 * \par Example
 * \include Matrices/SparseMatrix/SparseMatrixExample_getRow.cpp
 * \par Output
 * \include SparseMatrixExample_getRow.out
 *
 * \par Example
 * \include Matrices/SparseMatrix/SparseMatrixViewExample_getRow.cpp
 * \par Output
 * \include SparseMatrixViewExample_getRow.out
 */
template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
class SparseMatrixRowView
{
   public:

      /**
       * \brief Tells whether the parent matrix is a binary matrix.
       * @return `true` if the matrix is binary.
       */
      static constexpr bool isBinary() { return std::is_same< std::remove_const_t< RealType >, bool >::value; };

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename ValuesView::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename ColumnsIndexesView::IndexType;

      /**
       * \brief Type representing matrix row format.
       */
      using SegmentViewType = SegmentView;

      /**
       * \brief Type of container view used for storing the matrix elements values.
       */
      using ValuesViewType = ValuesView;

      /**
       * \brief Type of container view used for storing the column indexes of the matrix elements.
       */
      using ColumnsIndexesViewType = ColumnsIndexesView;

      /**
       * \brief Type of constant container view used for storing the matrix elements values.
       */
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;

      /**
       * \brief Type of constant container view used for storing the column indexes of the matrix elements.
       */
      using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;

      /**
       * \brief Type of sparse matrix row view.
       */
      using RowView = SparseMatrixRowView< SegmentView, ValuesViewType, ColumnsIndexesViewType >;

      /**
       * \brief Type of constant sparse matrix row view.
       */
      using ConstView = SparseMatrixRowView< SegmentView, ConstValuesViewType, ConstColumnsIndexesViewType >;

      /**
       * \brief The type of related matrix element.
       */
      using MatrixElementType = SparseMatrixElement< RealType, IndexType >;

      /**
       * \brief Type of iterator for the matrix row.
       */
      using IteratorType = MatrixRowViewIterator< RowView >;

      using ValueGetterType = details::SparseMatrixRowViewValueGetter< SegmentView, ValuesView, ColumnsIndexesView >;

      /**
       * \brief Constructor with \e segmentView, \e values and \e columnIndexes.
       *
       * \param segmentView instance of SegmentViewType representing matrix row.
       * \param values is a container view for storing the matrix elements values.
       * \param columnIndexes is a container view for storing the column indexes of the matrix elements.
       */
      __cuda_callable__
      SparseMatrixRowView( const SegmentViewType& segmentView,
                           const ValuesViewType& values,
                           const ColumnsIndexesViewType& columnIndexes );

      /**
       * \brief Returns size of the matrix row, i.e. number of matrix elements in this row.
       *
       * \return Size of the matrix row.
       */
      __cuda_callable__
      IndexType getSize() const;

      /**
       * \brief Returns the matrix row index.
       *
       * \return matrix row index.
       */
      __cuda_callable__
      const IndexType& getRowIndex() const;

      /**
       * \brief Returns constants reference to a column index of an element with given rank in the row.
       *
       * \param localIdx is the rank of the non-zero element in given row.
       *
       * \return constant reference to the matrix element column index.
       */
      __cuda_callable__
      const IndexType& getColumnIndex( const IndexType localIdx ) const;

      /**
       * \brief Returns non-constants reference to a column index of an element with given rank in the row.
       *
       * \param localIdx is the rank of the non-zero element in given row.
       *
       * \return non-constant reference to the matrix element column index.
       */
      __cuda_callable__
      IndexType& getColumnIndex( const IndexType localIdx );

      /**
       * \brief Returns constants reference to value of an element with given rank in the row.
       *
       * \param localIdx is the rank of the non-zero element in given row.
       *
       * \return constant reference to the matrix element value.
       */
      __cuda_callable__
      auto getValue( const IndexType localIdx ) const -> typename ValueGetterType::ConstResultType;

      /**
       * \brief Returns non-constants reference to value of an element with given rank in the row.
       *
       * \param localIdx is the rank of the non-zero element in given row.
       *
       * \return non-constant reference to the matrix element value.
       */
      __cuda_callable__
      auto getValue( const IndexType localIdx ) -> typename ValueGetterType::ResultType;

      /**
       * \brief Sets a value of matrix element with given rank in the matrix row.
       *
       * \param localIdx is the rank of the matrix element in the row.
       * \param value is the new value of the matrix element.
       */
      __cuda_callable__
      void setValue( const IndexType localIdx,
                     const RealType& value );

      /**
       * \brief Sets a column index of matrix element with given rank in the matrix row.
       *
       * \param localIdx is the rank of the matrix element in the row.
       * \param columnIndex is the new column index of the matrix element.
       */
      __cuda_callable__
      void setColumnIndex( const IndexType localIdx,
                           const IndexType& columnIndex );

      /**
       * \brief Sets both a value and a column index of matrix element with given rank in the matrix row.
       *
       * \param localIdx is the rank of the matrix element in the row.
       * \param columnIndex is the new column index of the matrix element.
       * \param value is the new value of the matrix element.
       */
      __cuda_callable__
      void setElement( const IndexType localIdx,
                       const IndexType columnIndex,
                       const RealType& value );

      /**
       * \brief Comparison of two matrix rows.
       *
       * The other matrix row can be from any other matrix.
       *
       * \param other is another matrix row.
       * \return \e true if both rows are the same, \e false otherwise.
       */
      template< typename _SegmentView,
                typename _ValuesView,
                typename _ColumnsIndexesView >
      __cuda_callable__
      bool operator==( const SparseMatrixRowView< _SegmentView, _ValuesView, _ColumnsIndexesView >& other ) const;

      /**
       * \brief Returns iterator pointing at the beginning of the matrix row.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      IteratorType begin();

      /**
       * \brief Returns iterator pointing at the end of the matrix row.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      IteratorType end();

      /**
       * \brief Returns constant iterator pointing at the beginning of the matrix row.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      const IteratorType cbegin() const;

      /**
       * \brief Returns constant iterator pointing at the end of the matrix row.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      const IteratorType cend() const;

      __cuda_callable__
      IndexType getPaddingIndex() const { return -1; };

   protected:

      SegmentViewType segmentView;

      ValuesViewType values;

      ColumnsIndexesViewType columnIndexes;
};

/**
 * \brief Insertion operator for a sparse matrix row.
 *
 * \param str is an output stream.
 * \param row is an input sparse matrix row.
 * \return  reference to the output stream.
 */
template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
std::ostream& operator<<( std::ostream& str, const SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >& row );

} // namespace Matrices
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Matrices/SparseMatrixRowView.hpp>
