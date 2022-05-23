// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/LambdaMatrixRowView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>

namespace noa::TNL {
namespace Matrices {

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::LambdaMatrixRowView(
   const MatrixElementsLambdaType& matrixElementsLambda,
   const CompressedRowLengthsLambdaType& compressedRowLengthsLambda,
   const IndexType& rows,
   const IndexType& columns,
   const IndexType& rowIdx )
: matrixElementsLambda( matrixElementsLambda ), compressedRowLengthsLambda( compressedRowLengthsLambda ), rows( rows ),
  columns( columns ), rowIdx( rowIdx )
{}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::getSize() const -> IndexType
{
   return this->compressedRowLengthsLambda( this->rows, this->columns, this->rowIdx );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::getRowIndex() const -> const IndexType&
{
   return this->rowIdx;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::getColumnIndex(
   const IndexType localIdx ) const -> IndexType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   RealType value;
   IndexType columnIdx;
   this->matrixElementsLambda( this->rows, this->columns, this->rowIdx, localIdx, columnIdx, value );
   return columnIdx;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::getValue( const IndexType localIdx ) const
   -> RealType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   RealType value;
   IndexType columnIdx;
   this->matrixElementsLambda( this->rows, this->columns, this->rowIdx, localIdx, columnIdx, value );
   return value;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
template< typename MatrixElementsLambda_, typename CompressedRowLengthsLambda_, typename Real_, typename Index_ >
__cuda_callable__
bool
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::operator==(
   const LambdaMatrixRowView< MatrixElementsLambda_, CompressedRowLengthsLambda_, Real_, Index_ >& other ) const
{
   IndexType i = 0;
   while( i < getSize() && i < other.getSize() ) {
      if( getColumnIndex( i ) != other.getColumnIndex( i ) )
         return false;
      ++i;
   }
   for( IndexType j = i; j < getSize(); j++ )
      // TODO: use ... != getPaddingIndex()
      if( getColumnIndex( j ) >= 0 )
         return false;
   for( IndexType j = i; j < other.getSize(); j++ )
      // TODO: use ... != getPaddingIndex()
      if( other.getColumnIndex( j ) >= 0 )
         return false;
   return true;
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::begin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::end() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
__cuda_callable__
auto
LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename MatrixElementsLambda, typename CompressedRowLengthsLambda, typename Real, typename Index >
std::ostream&
operator<<( std::ostream& str, const LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >& row )
{
   using NonConstIndex = std::remove_const_t<
      typename LambdaMatrixRowView< MatrixElementsLambda, CompressedRowLengthsLambda, Real, Index >::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}

}  // namespace Matrices
}  // namespace noa::TNL
