// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/ElementsOrganization.h>

namespace noa::TNL {
namespace Matrices {
namespace details {

template< typename Index, Algorithms::Segments::ElementsOrganization Organization >
class TridiagonalMatrixIndexer
{
public:
   using IndexType = Index;
   using ConstType = TridiagonalMatrixIndexer< std::add_const_t< Index >, Organization >;

   static constexpr bool
   getRowMajorOrder()
   {
      return Organization == Algorithms::Segments::RowMajorOrder;
   }

   __cuda_callable__
   TridiagonalMatrixIndexer() : rows( 0 ), columns( 0 ), nonemptyRows( 0 ) {}

   __cuda_callable__
   TridiagonalMatrixIndexer( const IndexType& rows, const IndexType& columns )
   : rows( rows ), columns( columns ), nonemptyRows( TNL::min( rows, columns ) + ( rows > columns ) )
   {}

   __cuda_callable__
   TridiagonalMatrixIndexer( const TridiagonalMatrixIndexer& indexer )
   : rows( indexer.rows ), columns( indexer.columns ), nonemptyRows( indexer.nonemptyRows )
   {}

   void
   setDimensions( const IndexType& rows, const IndexType& columns )
   {
      this->rows = rows;
      this->columns = columns;
      this->nonemptyRows = min( rows, columns ) + ( rows > columns );
   }

   __cuda_callable__
   IndexType
   getRowSize( const IndexType rowIdx ) const
   {
      return 3;
   }

   __cuda_callable__
   const IndexType&
   getRows() const
   {
      return this->rows;
   }

   __cuda_callable__
   const IndexType&
   getColumns() const
   {
      return this->columns;
   }

   __cuda_callable__
   const IndexType&
   getNonemptyRowsCount() const
   {
      return this->nonemptyRows;
   }

   __cuda_callable__
   IndexType
   getStorageSize() const
   {
      return 3 * this->nonemptyRows;
   }

   __cuda_callable__
   IndexType
   getGlobalIndex( const Index rowIdx, const Index localIdx ) const
   {
      TNL_ASSERT_GE( localIdx, 0, "" );
      TNL_ASSERT_LT( localIdx, 3, "" );
      TNL_ASSERT_GE( rowIdx, 0, "" );
      TNL_ASSERT_LT( rowIdx, this->rows, "" );

      if( Organization == Algorithms::Segments::RowMajorOrder )
         return 3 * rowIdx + localIdx;
      else
         return localIdx * nonemptyRows + rowIdx;
   }

protected:
   IndexType rows, columns, nonemptyRows;
};

}  // namespace details
}  // namespace Matrices
}  // namespace noa::TNL
