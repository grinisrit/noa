// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Matrices {

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
MultidiagonalMatrixRowView( const IndexType rowIdx,
                            const DiagonalsOffsetsView& diagonalsOffsets,
                            const ValuesViewType& values,
                            const IndexerType& indexer )
: rowIdx( rowIdx ), diagonalsOffsets( diagonalsOffsets ), values( values ), indexer( indexer )
{
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getSize() const -> IndexType
{
   return diagonalsOffsets.getSize();//indexer.getRowSize( rowIdx );
}


template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getRowIndex() const -> const IndexType&
{
   return this->rowIdx;
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getColumnIndex( const IndexType localIdx ) const -> const IndexType
{
   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, indexer.getDiagonals(), "" );
   return rowIdx + diagonalsOffsets[ localIdx ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getValue( const IndexType localIdx ) const -> const RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
getValue( const IndexType localIdx ) -> RealType&
{
   return this->values[ this->indexer.getGlobalIndex( rowIdx, localIdx ) ];
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
void
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
setElement( const IndexType localIdx,
            const RealType& value )
{
   this->values[ indexer.getGlobalIndex( rowIdx, localIdx ) ] = value;
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
begin() -> IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
end() -> IteratorType
{
   return IteratorType( *this, this->getSize() );
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
cbegin() const -> const IteratorType
{
   return IteratorType( *this, 0 );
}

template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView >
__cuda_callable__
auto
MultidiagonalMatrixRowView< ValuesView, Indexer, DiagonalsOffsetsView >::
cend() const -> const IteratorType
{
   return IteratorType( *this, this->getSize() );
}

} // namespace Matrices
} // namespace noaTNL
