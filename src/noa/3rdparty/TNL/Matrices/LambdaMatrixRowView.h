// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Matrices/LambdaMatrixRowViewIterator.h>
#include <TNL/Matrices/LambdaMatrixElement.h>


namespace TNL {
namespace Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of Lambda matrix.
 *
 * \tparam MatrixElementsLambda is a lambda function returning matrix elements values and positions.
 *
 * It has the following form:
 *
 * ```
 * auto matrixElements = [] __cuda_callable__ ( Index rows, Index columns, Index rowIdx, Index localIdx, Index& columnIdx, Real& value ) { ... }
 * ```
 *
 *    where \e rows is the number of matrix rows, \e columns is the number of matrix columns, \e rowIdx is the index of matrix row being queried,
 *    \e localIdx is the rank of the non-zero element in given row, \e columnIdx is a column index of the matrix element computed by
 *    this lambda and \e value is a value of the matrix element computed by this lambda.
 * \tparam CompressedRowLengthsLambda is a lambda function returning a number of non-zero elements in each row.
 *
 * It has the following form:
 *
 * ```
 * auto rowLengths = [] __cuda_callable__ ( Index rows, Index columns, Index rowIdx ) -> IndexType { ... }
 * ```
 *
 *    where \e rows is the number of matrix rows, \e columns is the number of matrix columns and \e rowIdx is an index of the row being queried.
 *
 * \tparam Real is a type of matrix elements values.
 * \tparam Index is a type to be used for indexing.
 *
 * \par Example
 * \include Matrices/LambdaMatrix/LambdaMatrixExample_getRow.cpp
 * \par Output
 * \include LambdaMatrixExample_getRow.out
 */
template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real = double,
          typename Index = int >
class LambdaMatrixRowView
{
   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = Real;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Type of the lambda function returning the matrix elements.
       */
      using MatrixElementsLambdaType = MatrixElementsLambda;

      /**
       * \brief Type of the lambda function returning the number of non-zero elements in each row.
       */
      using CompressedRowLengthsLambdaType = CompressedRowLengthsLambda;

      /**
       * \brief Type of Lambda matrix row view.
       */
      using RowView = LambdaMatrixRowView< MatrixElementsLambdaType, CompressedRowLengthsLambdaType, RealType, IndexType >;

      /**
       * \brief Type of constant Lambda matrix row view.
       */
      using ConstRowView = RowView;

      /**
       * \brief The type of related matrix element.
       */
      using MatrixElementType = LambdaMatrixElement< RealType, IndexType >;

      /**
       * \brief Type of iterator for the matrix row.
       */
      using IteratorType = LambdaMatrixRowViewIterator< RowView >;

      /**
       * \brief Constructor with related lambda functions, matrix dimensions and row index.
       *
       * \param matrixElementsLambda is a constant reference to the lambda function evaluating matrix elements.
       * \param compressedRowLengthsLambda is a constant reference to the lambda function returning the number of nonzero elements in each row.
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param rowIdx is the matrix row index.
       */
      __cuda_callable__
      LambdaMatrixRowView( const MatrixElementsLambdaType& matrixElementsLambda,
                           const CompressedRowLengthsLambdaType& compressedRowLengthsLambda,
                           const IndexType& rows,
                           const IndexType& columns,
                           const IndexType& rowIdx );

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
      IndexType getColumnIndex( const IndexType localIdx ) const;

      /**
       * \brief Returns constants reference to value of an element with given rank in the row.
       *
       * \param localIdx is the rank of the non-zero element in given row.
       *
       * \return constant reference to the matrix element value.
       */
      __cuda_callable__
      RealType getValue( const IndexType localIdx ) const;

      /**
       * \brief Comparison of two matrix rows.
       *
       * The other matrix row can be from any other matrix.
       *
       * \param other is another matrix row.
       * \return \e true if both rows are the same, \e false otherwise.
       */
      template< typename MatrixElementsLambda_,
                typename CompressedRowLengthsLambda_,
                typename Real_,
                typename Index_ >
      __cuda_callable__
      bool operator==( const LambdaMatrixRowView< MatrixElementsLambda_, CompressedRowLengthsLambda_, Real_, Index_ >& other ) const;

      /**
       * \brief Returns non-constant iterator pointing at the beginning of the matrix row.
       *
       * \return iterator pointing at the beginning.
       */
      __cuda_callable__
      const IteratorType begin() const;

      /**
       * \brief Returns non-constant iterator pointing at the end of the matrix row.
       *
       * \return iterator pointing at the end.
       */
      __cuda_callable__
      const IteratorType end() const;

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

      const MatrixElementsLambda& matrixElementsLambda;

      const CompressedRowLengthsLambda& compressedRowLengthsLambda;

      IndexType rows, columns, rowIdx;
};

/**
 * \brief Insertion operator for a Lambda matrix row.
 *
 * \param str is an output stream.
 * \param row is an input Lambda matrix row.
 * \return  reference to the output stream.
 */
template< typename MatrixElementsLambda,
          typename CompressedRowLengthsLambda,
          typename Real,
          typename Index >
std::ostream& operator<<( std::ostream& str, const LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >& row );

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/LambdaMatrixRowView.hpp>
