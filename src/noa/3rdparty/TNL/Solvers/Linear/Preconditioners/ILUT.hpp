// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <vector>
#include <set>

#include "ILUT.h"
#include <TNL/Solvers/Linear/Utils/TriangularSolve.h>
#include <TNL/Timer.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix, typename Real, typename Index >
bool
ILUT_impl< Matrix, Real, Devices::Host, Index >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "ilut-p" ) )
      p = parameters.getParameter< int >( prefix + "ilut-p" );
   if( parameters.checkParameter( prefix + "ilut-threshold" ) )
      tau = parameters.getParameter< double >( prefix + "ilut-threshold" );
   return true;
}

template< typename Matrix, typename Real, typename Index >
void
ILUT_impl< Matrix, Real, Devices::Host, Index >::
update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   const auto& localMatrix = Traits< Matrix >::getLocalMatrix( *matrixPointer );
   const IndexType N = localMatrix.getRows();
   const IndexType minColumn = getMinColumn( *matrixPointer );

   L.setDimensions( N, N );
   U.setDimensions( N, N );

//   Timer timer_total, timer_rowlengths, timer_copy_into_w, timer_k_loop, timer_heap_construct, timer_heap_extract, timer_copy_into_LU, timer_reset;

//   timer_total.start();

   // compute row lengths
//   timer_rowlengths.start();
   typename decltype(L)::RowsCapacitiesType L_rowLengths( N );
   typename decltype(U)::RowsCapacitiesType U_rowLengths( N );
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
      // store p additional entries in each factor
      L_rowLengths[ i ] = L_entries + p;
      U_rowLengths[ N - 1 - i ] = U_entries + p;
   }
   L.setRowCapacities( L_rowLengths );
   U.setRowCapacities( U_rowLengths );
//   timer_rowlengths.stop();

   // intermediate full vector for the i-th row of A
   VectorType w;
   w.setSize( N );
   w.setValue( 0.0 );

   // intermediate vectors for sorting and keeping only the largest values
   struct Triplet {
      IndexType column;
      RealType value;
      RealType abs_value;
      Triplet(IndexType column, RealType value, RealType abs_value) : column(column), value(value), abs_value(abs_value) {}
   };
   auto cmp_abs_value = []( const Triplet& a, const Triplet& b ){ return a.abs_value < b.abs_value; };
   std::vector< Triplet > heap_L, heap_U;
   auto cmp_column = []( const Triplet& a, const Triplet& b ){ return a.column < b.column; };
   std::vector< Triplet > values_L, values_U;

   // Incomplete LU factorization with threshold
   // (see Saad - Iterative methods for sparse linear systems, section 10.4)
   for( IndexType i = 0; i < N; i++ ) {
      const auto A_i = localMatrix.getRow( i );

      RealType A_i_norm = 0.0;

      // set of indices where w_k is non-zero (i.e. {k: w_k != 0})
      std::set< IndexType > w_k_set;

      // copy A_i into the full vector w
//      timer_copy_into_w.start();
      for( IndexType c_j = 0; c_j < A_i.getSize(); c_j++ ) {
         auto j = A_i.getColumnIndex( c_j );
         if( minColumn > 0 ) {
            // skip non-local elements
            if( j < minColumn ) continue;
            j -= minColumn;
         }
         // handle ellpack dummy entries
         if( j == localMatrix.getPaddingIndex() ) break;
         w[ j ] = A_i.getValue( c_j );

         // running computation of norm
         A_i_norm += w[ j ] * w[ j ];

         w_k_set.insert( j );
      }
//      timer_copy_into_w.stop();

      // compute relative tolerance
      A_i_norm = std::sqrt( A_i_norm );
      const RealType tau_i = tau * A_i_norm;

      // loop for k = 0, ..., i - 1; but only over the non-zero entries of w
//      timer_k_loop.start();
      for( const IndexType k : w_k_set ) {
         if( k >= i )
            break;

         RealType w_k = w[ k ] / localMatrix.getElement( k, k + minColumn );

         // apply dropping rule to w_k
         if( std::abs( w_k ) < tau_i )
            w_k = 0.0;

         w[ k ] = w_k;

         if( w_k != 0.0 ) {
            // w := w - w_k * U_k
            const auto U_k = U.getRow( N - 1 - k );
            // loop for j = 0, ..., N-1; but only over the non-zero entries
            for( Index c_j = 0; c_j < U_rowLengths[ N - 1 - k ]; c_j++ ) {
               const auto j = U_k.getColumnIndex( c_j );

               // skip dropped entries
               if( j == localMatrix.getPaddingIndex() ) break;
               w[ j ] -= w_k * U_k.getValue( c_j );

               // add non-zero to the w_k_set
               w_k_set.insert( j );
            }
         }
      }
//      timer_k_loop.stop();

      // apply dropping rule to the row w
      // (we drop all values under threshold and keep nl(i) + p largest values in L
      // and nu(i) + p largest values in U; see Saad (2003) for reference)

      // construct heaps with the values in the L and U parts separately
//      timer_heap_construct.start();
      for( const IndexType j : w_k_set ) {
         const RealType w_j_abs = std::abs( w[ j ] );
         // ignore small values
         if( w_j_abs < tau_i )
            continue;
         // push into the heaps for L or U
         if( j < i )
            heap_L.push_back( Triplet( j, w[ j ], w_j_abs ) );
         else
            heap_U.push_back( Triplet( j, w[ j ], w_j_abs ) );
      }
      std::make_heap( heap_L.begin(), heap_L.end(), cmp_abs_value );
      std::make_heap( heap_U.begin(), heap_U.end(), cmp_abs_value );
//      timer_heap_construct.stop();

      // extract values for L and U
//      timer_heap_extract.start();
      for( IndexType c_j = 0; c_j < L_rowLengths[ i ] && c_j < (IndexType) heap_L.size(); c_j++ ) {
         // move the largest to the end
         std::pop_heap( heap_L.begin(), heap_L.end(), cmp_abs_value );
         // move the triplet from one vector into another
         const auto largest = heap_L.back();
         heap_L.pop_back();
         values_L.push_back( largest );
      }
      for( IndexType c_j = 0; c_j < U_rowLengths[ N - 1 - i ] && c_j < (IndexType) heap_U.size(); c_j++ ) {
         // move the largest to the end
         std::pop_heap( heap_U.begin(), heap_U.end(), cmp_abs_value );
         // move the triplet from one vector into another
         const auto largest = heap_U.back();
         heap_U.pop_back();
         values_U.push_back( largest );
      }
//      timer_heap_extract.stop();

//      std::cout << "i = " << i << ", L_rowLengths[ i ] = " << L_rowLengths[ i ] << ", U_rowLengths[ i ] = " << U_rowLengths[ N - 1 - i ] << std::endl;

//      timer_copy_into_LU.start();

      // sort by column index to make it insertable into the sparse matrix
      std::sort( values_L.begin(), values_L.end(), cmp_column );
      std::sort( values_U.begin(), values_U.end(), cmp_column );

      // the row L_i might be empty
      if( values_L.size() ) {
         // L_ij = w_j for j = 0, ..., i - 1
         auto L_i = L.getRow( i );
         for( IndexType c_j = 0; c_j < (IndexType) values_L.size(); c_j++ ) {
            const auto j = values_L[ c_j ].column;
            L_i.setElement( c_j, j, values_L[ c_j ].value );
         }
      }

      // U_ij = w_j for j = i, ..., N - 1
      auto U_i = U.getRow( N - 1 - i );
      for( IndexType c_j = 0; c_j < (IndexType) values_U.size(); c_j++ ) {
         const auto j = values_U[ c_j ].column;
         U_i.setElement( c_j, j, values_U[ c_j ].value );
      }

//      timer_copy_into_LU.stop();

      // reset w
//      timer_reset.start();
      for( const IndexType j : w_k_set )
         w[ j ] = 0.0;

      heap_L.clear();
      heap_U.clear();
      values_L.clear();
      values_U.clear();
//      timer_reset.stop();
   }

//   timer_total.stop();

//   std::cout << "ILUT::update statistics:\n";
//   std::cout << "\ttimer_total:           " << timer_total.getRealTime()          << " s\n";
//   std::cout << "\ttimer_rowlengths:      " << timer_rowlengths.getRealTime()     << " s\n";
//   std::cout << "\ttimer_copy_into_w:     " << timer_copy_into_w.getRealTime()    << " s\n";
//   std::cout << "\ttimer_k_loop:          " << timer_k_loop.getRealTime()         << " s\n";
//   std::cout << "\ttimer_heap_construct:  " << timer_heap_construct.getRealTime() << " s\n";
//   std::cout << "\ttimer_heap_extract:    " << timer_heap_extract.getRealTime()   << " s\n";
//   std::cout << "\ttimer_copy_into_LU:    " << timer_copy_into_LU.getRealTime()   << " s\n";
//   std::cout << "\ttimer_reset:           " << timer_reset.getRealTime()          << " s\n";
//   std::cout << std::flush;
}

template< typename Matrix, typename Real, typename Index >
void
ILUT_impl< Matrix, Real, Devices::Host, Index >::
solve( ConstVectorViewType _b, VectorViewType _x ) const
{
   const auto b = Traits< Matrix >::getConstLocalView( _b );
   auto x = Traits< Matrix >::getLocalView( _x );

   // Step 1: solve y from Ly = b
   triangularSolveLower< false >( L, x, b );

   // Step 2: solve x from Ux = y
   triangularSolveUpper< true, false >( U, x, x );

   // synchronize ghosts
   Traits< Matrix >::startSynchronization( _x );
}

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
