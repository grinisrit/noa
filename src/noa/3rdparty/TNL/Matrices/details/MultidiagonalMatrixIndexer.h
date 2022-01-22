// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
   namespace Matrices {
      namespace details {

template< typename Index,
          bool RowMajorOrder >
class MultidiagonalMatrixIndexer
{
   public:

      using IndexType = Index;
      using ConstType = MultidiagonalMatrixIndexer< std::add_const_t< Index >, RowMajorOrder >;

      static constexpr bool getRowMajorOrder() { return RowMajorOrder; };

      __cuda_callable__
      MultidiagonalMatrixIndexer()
      : rows( 0 ), columns( 0 ), nonemptyRows( 0 ){};

      __cuda_callable__
      MultidiagonalMatrixIndexer( const IndexType& rows,
                                  const IndexType& columns,
                                  const IndexType& diagonals,
                                  const IndexType& nonemptyRows )
      : rows( rows ),
        columns( columns ),
        diagonals( diagonals ),
        nonemptyRows( nonemptyRows ) {};

      __cuda_callable__
      MultidiagonalMatrixIndexer( const MultidiagonalMatrixIndexer& indexer )
      : rows( indexer.rows ),
        columns( indexer.columns ),
        diagonals( indexer.diagonals ),
        nonemptyRows( indexer.nonemptyRows ) {};

      void set( const IndexType& rows,
                const IndexType& columns,
                const IndexType& diagonals,
                const IndexType& nonemptyRows )
      {
         this->rows = rows;
         this->columns = columns;
         this->diagonals = diagonals;
         this->nonemptyRows = nonemptyRows;
      };

      __cuda_callable__
      const IndexType& getRows() const { return this->rows; };

      __cuda_callable__
      const IndexType& getColumns() const { return this->columns; };

      __cuda_callable__
      const IndexType& getDiagonals() const { return this->diagonals; };

      __cuda_callable__
      const IndexType& getNonemptyRowsCount() const { return this->nonemptyRows; };

      __cuda_callable__
      IndexType getStorageSize() const { return diagonals * this->nonemptyRows; };

      __cuda_callable__
      IndexType getGlobalIndex( const Index rowIdx, const Index localIdx ) const
      {
         TNL_ASSERT_GE( localIdx, 0, "" );
         TNL_ASSERT_LT( localIdx, diagonals, "" );
         TNL_ASSERT_GE( rowIdx, 0, "" );
         TNL_ASSERT_LT( rowIdx, this->rows, "" );

         if( RowMajorOrder )
            return diagonals * rowIdx + localIdx;
         else
            return localIdx * nonemptyRows + rowIdx;
      };

      protected:

         IndexType rows, columns, diagonals, nonemptyRows;
};
      } //namespace details
   } // namespace Matrices
} // namespace noaTNL
