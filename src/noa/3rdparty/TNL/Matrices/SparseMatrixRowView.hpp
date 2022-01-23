// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Matrices/SparseMatrixRowView.h>
#include <noa/3rdparty/TNL/Assert.h>

namespace noa::TNL {
namespace Matrices {

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
SparseMatrixRowView( const SegmentViewType& segmentView,
                     const ValuesViewType& values,
                     const ColumnsIndexesViewType& columnIndexes )
 : segmentView( segmentView ), values( values ), columnIndexes( columnIndexes )
{
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getSize() const -> IndexType
{
   return segmentView.getSize();
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__
auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getRowIndex() const -> const IndexType&
{
   return segmentView.getSegmentIndex();
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getColumnIndex( const IndexType localIdx ) -> IndexType&
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return columnIndexes[ segmentView.getGlobalIndex( localIdx ) ];
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getValue( const IndexType localIdx ) const -> typename ValueGetterType::ConstResultType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return ValueGetterType::getValue( segmentView.getGlobalIndex( localIdx ),
                                     values,
                                     columnIndexes,
                                     this->getPaddingIndex() );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
getValue( const IndexType localIdx ) -> typename ValueGetterType::ResultType
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   return ValueGetterType::getValue( segmentView.getGlobalIndex( localIdx ),
                                     values,
                                     columnIndexes,
                                     this->getPaddingIndex() );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
setValue( const IndexType localIdx,
          const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   if( ! isBinary() ) {
      const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
      values[ globalIdx ] = value;
   }
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
setColumnIndex( const IndexType localIdx,
                const IndexType& columnIndex )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   this->columnIndexes[ globalIdx ] = columnIndex;
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ void
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
setElement( const IndexType localIdx,
            const IndexType column,
            const RealType& value )
{
   TNL_ASSERT_LT( localIdx, this->getSize(), "Local index exceeds matrix row capacity." );
   const IndexType globalIdx = segmentView.getGlobalIndex( localIdx );
   columnIndexes[ globalIdx ] = column;
   if( ! isBinary() )
      values[ globalIdx ] = value;
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
   template< typename _SegmentView,
             typename _ValuesView,
             typename _ColumnsIndexesView >
__cuda_callable__
bool
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
operator==( const SparseMatrixRowView< _SegmentView, _ValuesView, _ColumnsIndexesView >& other ) const
{
   IndexType i = 0;
   while( i < getSize() && i < other.getSize() ) {
      if( getColumnIndex( i ) != other.getColumnIndex( i ) )
         return false;
      if( ! isBinary() && getValue( i ) != other.getValue( i ) )
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

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
begin() -> IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
end() -> IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
__cuda_callable__ auto
SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::
cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView >
std::ostream& operator<<( std::ostream& str, const SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >& row )
{
   using NonConstIndex = std::remove_const_t< typename SparseMatrixRowView< SegmentView, ValuesView, ColumnsIndexesView >::IndexType >;
   for( NonConstIndex i = 0; i < row.getSize(); i++ )
      if( row.isBinary() )
         // TODO: check getPaddingIndex(), print only the column indices of non-zeros but not the values
         str << " [ " << row.getColumnIndex( i ) << " ] = " << (row.getColumnIndex( i ) >= 0) << ", ";
      else
         str << " [ " << row.getColumnIndex( i ) << " ] = " << row.getValue( i ) << ", ";
   return str;
}


} // namespace Matrices
} // namespace noa::TNL
