// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/LambdaMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/LambdaMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/details/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::LambdaMatrix(
   MatrixElementsLambda& matrixElements,
   CompressedRowLengthsLambda& compressedRowLengths )
: rows( 0 ), columns( 0 ), matrixElementsLambda( matrixElements ), compressedRowLengthsLambda( compressedRowLengths )
{}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::LambdaMatrix(
   const IndexType& rows,
   const IndexType& columns,
   MatrixElementsLambda& matrixElements,
   CompressedRowLengthsLambda& compressedRowLengths )
: rows( rows ), columns( columns ), matrixElementsLambda( matrixElements ), compressedRowLengthsLambda( compressedRowLengths )
{}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::setDimensions( const IndexType& rows,
                                                                                                      const IndexType& columns )
{
   this->rows = rows;
   this->columns = columns;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
__cuda_callable__
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
__cuda_callable__
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
__cuda_callable__
const CompressedRowLengthsLambda&
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getCompressedRowLengthsLambda() const
{
   return this->compressedRowLengthsLambda;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
__cuda_callable__
const MatrixElementsLambda&
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getMatrixElementsLambda() const
{
   return this->matrixElementsLambda;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Vector >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getRowCapacities(
   Vector& rowCapacities ) const
{
   this->getCompressedRowLengths( rowCapacities );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Vector >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getCompressedRowLengths(
   Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
Index
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getNonzeroElementsCount() const
{
   Containers::Vector< IndexType, DeviceType, IndexType > rowLengthsVector;
   this->getCompressedRowLengths( rowLengthsVector );
   return sum( rowLengthsVector );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
Real
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getElement(
   const IndexType row,
   const IndexType column ) const
{
   Containers::Array< RealType, DeviceType > value( 1 );
   auto valueView = value.getView();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto getValue = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const IndexType rowSize = rowLengths( rows, columns, row );
      valueView[ 0 ] = 0.0;
      for( IndexType localIdx = 0; localIdx < rowSize; localIdx++ ) {
         RealType elementValue;
         IndexType elementColumn;
         matrixElements( rows, columns, row, localIdx, elementColumn, elementValue );
         if( elementColumn == column ) {
            valueView[ 0 ] = elementValue;
            break;
         }
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( row, row + 1, getValue );
   return valueView.getElement( 0 );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
__cuda_callable__
auto
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::getRow( const IndexType& rowIdx ) const
   -> const RowView
{
   return RowView(
      this->getMatrixElementsLambda(), this->getCompressedRowLengthsLambda(), this->getRows(), this->getColumns(), rowIdx );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename InVector, typename OutVector >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   const RealType& matrixMultiplicator,
   const RealType& outVectorMultiplicator,
   const IndexType begin,
   IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   auto fetch = [ = ] __cuda_callable__( IndexType row, IndexType columnIdx, const RealType& value ) mutable -> RealType
   {
      if( value == 0.0 )
         return 0.0;
      return value * inVectorView[ columnIdx ];
   };
   auto reduce = [] __cuda_callable__( RealType & sum, const RealType& value ) -> RealType
   {
      return sum + value;
   };
   auto keep = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      if( outVectorMultiplicator == 0.0 )
         outVectorView[ row ] = matrixMultiplicator * value;
      else
         outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
   };
   if( ! end )
      end = this->getRows();
   this->reduceRows( begin, end, fetch, reduce, keep, 0.0 );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::reduceRows(
   IndexType first,
   IndexType last,
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchReal& identity ) const
{
   using FetchType = decltype( fetch( IndexType(), IndexType(), RealType() ) );

   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   auto processRow = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const IndexType rowLength = rowLengths( rows, columns, rowIdx );
      FetchType result = identity;
      for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
         IndexType elementColumn( 0 );
         RealType elementValue( 0.0 );
         matrixElements( rows, columns, rowIdx, localIdx, elementColumn, elementValue );
         FetchType fetchValue = identity;
         if( elementValue != 0.0 )
            fetchValue = fetch( rowIdx, elementColumn, elementValue );
         result = reduce( result, fetchValue );
      }
      keep( rowIdx, result );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, processRow );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::reduceAllRows(
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::forElements( IndexType first,
                                                                                                    IndexType last,
                                                                                                    Function& function ) const
{
   const IndexType rows = this->getRows();
   const IndexType columns = this->getColumns();
   auto rowLengths = this->compressedRowLengthsLambda;
   auto matrixElements = this->matrixElementsLambda;
   auto processRow = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const IndexType rowLength = rowLengths( rows, columns, rowIdx );
      for( IndexType localIdx = 0; localIdx < rowLength; localIdx++ ) {
         IndexType elementColumn( 0 );
         RealType elementValue( 0.0 );
         matrixElements( rows, columns, rowIdx, localIdx, elementColumn, elementValue );
         if( elementValue != 0.0 )
            function( rowIdx, localIdx, elementColumn, elementValue );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, processRow );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::forAllElements(
   Function& function ) const
{
   forElements( (IndexType) 0, this->getRows(), function );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::forRows( IndexType begin,
                                                                                                IndexType end,
                                                                                                Function&& function ) const
{
   auto view = *this;
   auto f = [ = ] __cuda_callable__( const IndexType rowIdx ) mutable
   {
      auto rowView = view.getRow( rowIdx );
      function( rowView );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::sequentialForRows(
   IndexType begin,
   IndexType end,
   Function&& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
template< typename Function >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::sequentialForAllRows(
   Function&& function ) const
{
   sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
void
LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ ) {
      str << "Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ ) {
         RealType value = this->getElement( row, column );
         if( value ) {
            std::stringstream str_;
            str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << value;
            str << std::setw( 10 ) << str_.str();
         }
      }
      str << std::endl;
   }
}

/**
 * \brief Insertion operator for dense matrix and output stream.
 *
 * \param str is the output stream.
 * \param matrix is the lambda matrix.
 * \return reference to the stream.
 */
template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str,
            const LambdaMatrix< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Device, Index >& matrix )
{
   matrix.print( str );
   return str;
}

}  // namespace Matrices
}  // namespace noa::TNL
