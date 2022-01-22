// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/TNL/Matrices/MatrixRowViewIterator.h>
#include <noa/3rdparty/TNL/Matrices/DenseMatrixElement.h>

namespace noaTNL {
   namespace Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of dense matrix.
 *
 * \tparam SegmentView is a segment view of segments representing the matrix format.
 * \tparam ValuesView is a vector view storing the matrix elements values.
 *
 * See \ref DenseMatrix and \ref DenseMatrixView.
 *
 * \par Example
 * \include Matrices/DenseMatrix/DenseMatrixExample_getRow.cpp
 * \par Output
 * \include DenseMatrixExample_getRow.out
 *
 * \par Example
 * \include Matrices/DenseMatrix/DenseMatrixViewExample_getRow.cpp
 * \par Output
 * \include DenseMatrixViewExample_getRow.out
 */
template< typename SegmentView,
          typename ValuesView >
class DenseMatrixRowView
{
   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename ValuesView::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename SegmentView::IndexType;

      /**
       * \brief Type representing matrix row format.
       */
      using SegmentViewType = SegmentView;

      /**
       * \brief Type of container view used for storing matrix elements values.
       */
      using ValuesViewType = ValuesView;

      /**
       * \brief Type of constant container view used for storing the matrix elements values.
       */
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;

      /**
       * \brief Type of dense matrix row view.
       */
      using RowView = DenseMatrixRowView< SegmentView, ValuesViewType >;

      /**
       * \brief Type of constant sparse matrix row view.
       */
      using ConstView = DenseMatrixRowView< SegmentView, ConstValuesViewType >;

      /**
       * \brief The type of related matrix element.
       */
      using MatrixElementType = DenseMatrixElement< RealType, IndexType >;

      /**
       * \brief Type of iterator for the matrix row.
       */
      using IteratorType = MatrixRowViewIterator< RowView >;

      /**
       * \brief Constructor with \e segmentView and \e values
       *
       * \param segmentView instance of SegmentViewType representing matrix row.
       * \param values is a container view for storing the matrix elements values.
       */
      __cuda_callable__
      DenseMatrixRowView( const SegmentViewType& segmentView,
                          const ValuesViewType& values );

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
       * \brief Returns constants reference to an element with given column index.
       *
       * \param column is column index of the matrix element.
       *
       * \return constant reference to the matrix element.
       */
      __cuda_callable__
      const RealType& getValue( const IndexType column ) const;

      /**
       * \brief Returns non-constants reference to an element with given column index.
       *
       * \param column is a column index of the matrix element.
       *
       * \return non-constant reference to the matrix element.
       */
      __cuda_callable__
      RealType& getValue( const IndexType column );

      /**
       * \brief This method is only for compatibility with sparse matrix row.
       *
       * \param localIdx is the rank of the matrix element in given row.
       *
       * \return the value of \ref localIdx as column index.
       */
      __cuda_callable__
      IndexType getColumnIndex( const IndexType localIdx ) const;

      /**
       * \brief Sets value of matrix element with given column index
       * .
       * \param column is a column index of the matrix element.
       * \param value is a value the matrix element will be set to.
       */
      __cuda_callable__
      void setValue( const IndexType column,
                     const RealType& value );

      /**
       * \brief Sets value of matrix element with given column index
       *
       * The \e localIdx parameter is here only for compatibility with
       * the sparse matrices and it is omitted.
       *
       * \param column is a column index of the matrix element.
       * \param value is a value the matrix element will be set to.
       */
      __cuda_callable__
      void setElement( const IndexType localIdx,
                       const IndexType column,
                       const RealType& value );

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

   protected:

      SegmentViewType segmentView;

      ValuesViewType values;
};
   } // namespace Matrices
} // namespace noaTNL

#include <noa/3rdparty/TNL/Matrices/DenseMatrixRowView.hpp>
