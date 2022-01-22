// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Matrices {

template< typename ValuesView, typename Indexer >
__cuda_callable__
TridiagonalMatrixRowView< ValuesView, Indexer >::
TridiagonalMatrixRowView( const IndexType rowIdx,
                          const ValuesViewType& values,
                          const IndexerType& indexer )
: rowIdx( rowIdx ), values( values ), indexer( indexer )
{
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
getSize() const -> IndexType
{
   return indexer.getRowSize( rowIdx );
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
getRowIndex() const -> const IndexType&
{
   return rowIdx;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType
{
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );
   return rowIdx + localIdx - 1;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
getValue( const IndexType localIdx ) -> RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
void
TridiagonalMatrixRowView< ValuesView, Indexer >::
setElement( const IndexType localIdx,
            const RealType& value )
{
   this->values[ indexer.getGlobalIndex( rowIdx, localIdx ) ] = value;
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
begin() -> IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
end() -> IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView, typename Indexer >
__cuda_callable__
auto
TridiagonalMatrixRowView< ValuesView, Indexer >::
cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

} // namespace Matrices
} // namespace noaTNL
