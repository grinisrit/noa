// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

#include "ILU0.h"
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/Utils/TriangularSolve.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix, typename Real, typename Index >
void
ILU0_impl< Matrix, Real, Devices::Host, Index >::update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   const auto& localMatrix = Traits< Matrix >::getLocalMatrix( *matrixPointer );
   const IndexType N = localMatrix.getRows();
   const IndexType minColumn = getMinColumn( *matrixPointer );

   L.setDimensions( N, N );
   U.setDimensions( N, N );

   // copy row lengths
   typename decltype( L )::RowsCapacitiesType L_rowLengths( N );
   typename decltype( U )::RowsCapacitiesType U_rowLengths( N );
   for( IndexType i = 0; i < N; i++ ) {
      const auto row = localMatrix.getRow( i );
      IndexType L_entries = 0;
      IndexType U_entries = 0;
      for( IndexType j = 0; j < row.getSize(); j++ ) {
         const auto column = row.getColumnIndex( j );
         if( column < minColumn )
            continue;
         if( column < i + minColumn )
            L_entries++;
         else if( column < N + minColumn )
            U_entries++;
         else
            break;
      }
      L_rowLengths[ i ] = L_entries;
      U_rowLengths[ N - 1 - i ] = U_entries;
   }
   L.setRowCapacities( L_rowLengths );
   U.setRowCapacities( U_rowLengths );

   // Incomplete LU factorization
   // The factors L and U are stored separately and the rows of U are reversed.
   for( IndexType i = 0; i < N; i++ ) {
      // copy all non-zero entries from A into L and U
      const auto row = localMatrix.getRow( i );
      const auto max_length = row.getSize();
      std::unique_ptr< IndexType[] > all_columns{ new IndexType[ max_length ] };
      std::unique_ptr< RealType[] > all_values{ new RealType[ max_length ] };
      for( IndexType j = 0; j < max_length; j++ ) {
         all_columns[ j ] = row.getColumnIndex( j );
         all_values[ j ] = row.getValue( j );
      }

      // skip non-local elements
      IndexType* columns = all_columns.get();
      RealType* values = all_values.get();
      while( columns[ 0 ] < minColumn ) {
         columns++;
         values++;
      }

      // update column indices
      if( minColumn > 0 )
         for( IndexType c_j = 0; c_j < max_length; c_j++ )
            all_columns[ c_j ] -= minColumn;

      const auto L_entries = L_rowLengths[ i ];
      const auto U_entries = U_rowLengths[ N - 1 - i ];
      //      L.setRow( i, columns, values, L_entries );
      //      U.setRow( N - 1 - i, &columns[ L_entries ], &values[ L_entries ], U_entries );

      // copy values into U
      auto U_i = U.getRow( N - 1 - i );
      for( IndexType c_j = 0; c_j < U_entries; c_j++ )
         U_i.setElement( c_j, columns[ L_entries + c_j ], values[ L_entries + c_j ] );

      // this condition is to avoid segfaults on empty L.getRow( i )
      if( L_entries > 0 ) {
         // copy values into L
         auto L_i = L.getRow( i );
         for( IndexType c_j = 0; c_j < L_entries; c_j++ )
            L_i.setElement( c_j, columns[ c_j ], values[ c_j ] );

         // loop for k = 0, ..., i - 1; but only over the non-zero entries
         for( IndexType c_k = 0; c_k < L_entries; c_k++ ) {
            const auto k = L_i.getColumnIndex( c_k );

            auto L_ik = L_i.getValue( c_k ) / U.getElement( N - 1 - k, k );
            L_i.setValue( c_k, L_ik );

            // loop for j = k+1, ..., N-1; but only over the non-zero entries
            // and split into two loops over L and U separately
            for( IndexType c_j = c_k + 1; c_j < L_entries; c_j++ ) {
               const auto j = L_i.getColumnIndex( c_j );
               const auto L_ij = L_i.getValue( c_j ) - L_ik * U.getElement( N - 1 - k, j );
               L_i.setValue( c_j, L_ij );
            }
            for( IndexType c_j = 0; c_j < U_entries; c_j++ ) {
               const auto j = U_i.getColumnIndex( c_j );
               const auto U_ij = U_i.getValue( c_j ) - L_ik * U.getElement( N - 1 - k, j );
               U_i.setValue( c_j, U_ij );
            }
         }
      }
   }
}

template< typename Matrix, typename Real, typename Index >
void
ILU0_impl< Matrix, Real, Devices::Host, Index >::solve( ConstVectorViewType _b, VectorViewType _x ) const
{
   const auto b = Traits< Matrix >::getConstLocalView( _b );
   auto x = Traits< Matrix >::getLocalView( _x );

   TNL_ASSERT_EQ( b.getSize(), L.getRows(), "The size of the vector b does not match the size of the decomposed matrix." );
   TNL_ASSERT_EQ( x.getSize(), U.getRows(), "The size of the vector x does not match the size of the decomposed matrix." );

   // Step 1: solve y from Ly = b
   triangularSolveLower< true >( L, x, b );

   // Step 2: solve x from Ux = y
   triangularSolveUpper< true, true >( U, x, x );

   // synchronize ghosts
   Traits< Matrix >::startSynchronization( _x );
}

}  // namespace Preconditioners
}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
