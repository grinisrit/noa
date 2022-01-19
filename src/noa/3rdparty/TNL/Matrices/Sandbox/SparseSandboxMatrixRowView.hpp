// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/Sandbox/SparseSandboxMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL {
   namespace Matrices {
      namespace Sandbox {

// SANDBOX_TODO: Modify the follwing constructor by your needs
template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
SparseSandboxMatrixRowView( IndexType rowIdx,
                            IndexType offset,
                            IndexType size,
                            const ValuesViewType& values,
                            const ColumnsIndexesViewType& columnIndexes )
 : rowIdx( rowIdx ), size( size ), offset( offset ), values( values ), columnIndexes( columnIndexes )
{
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__
auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getRowIndex() const -> const IndexType&
{
   return this->rowIdx;
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   return columnIndexes[ offset + localIdx ];
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getColumnIndex( const IndexType localIdx ) -> IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   return columnIndexes[ offset + localIdx ];
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   TNL_ASSERT_FALSE( isBinary(), "Cannot call this method for binary matrix row." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   return values[ offset + localIdx ];
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
getValue( const IndexType localIdx ) -> RealType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   TNL_ASSERT_FALSE( isBinary(), "Cannot call this method for binary matrix row." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   return values[ offset + localIdx ];
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
setValue( const IndexType localIdx,
          const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   if( ! isBinary() ) {
      // SANDBOX_TODO: Modify the following line to match with your sparse format.
      const IndexType globalIdx = offset + localIdx;
      values[ globalIdx ] = value;
   }
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
setColumnIndex( const IndexType localIdx,
                const IndexType& columnIndex )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   const IndexType globalIdx = offset + localIdx;
   this->columnIndexes[ globalIdx ] = columnIndex;
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ void
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
setElement( const IndexType localIdx,
            const IndexType column,
            const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   // SANDBOX_TODO: Modify the following line to match with your sparse format.
   const IndexType globalIdx = offset + localIdx;
   columnIndexes[ globalIdx ] = column;
   if( ! isBinary() )
      values[ globalIdx ] = value;
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
   template< typename _ValuesView,
             typename _ColumnsIndexesView,
             bool _isBinary >
__cuda_callable__
bool
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
operator==( const SparseSandboxMatrixRowView< _ValuesView, _ColumnsIndexesView, _isBinary >& other ) const
{
   IndexType i = 0;
   while( i < getSize() && i < other.getSize() ) {
      if( getColumnIndex( i ) != other.getColumnIndex( i ) )
         return false;
      if( ! _isBinary && getValue( i ) != other.getValue( i ) )
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

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
begin() -> IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
end() -> IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
__cuda_callable__ auto
SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::
cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename ValuesView,
          typename ColumnsIndexesView,
          bool isBinary_ >
std::ostream& operator<<( std::ostream& str, const SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >& row )
{
   using NonConstIndex = std::remove_const_t< typename SparseSandboxMatrixRowView< ValuesView, ColumnsIndexesView, isBinary_ >::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      if( isBinary_ )
         // TODO: check getPaddingIndex(), print only the column indices of non-zeros but not the values
         str << " [ " << row.getColumnIndex( i ) << " ] = " << (row.getColumnIndex( i ) >= 0) << ", ";
      else
         str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}

      } // namespace Sandbox
   } // namespace Matrices
} // namespace TNL
