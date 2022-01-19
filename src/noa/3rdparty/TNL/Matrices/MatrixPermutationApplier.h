// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Algorithms/ParallelFor.h>

namespace TNL {
namespace Matrices {

template< typename Matrix,
          typename PermutationArray >
void permuteMatrixRows( Matrix& matrix, const PermutationArray& perm )
{
   static_assert( std::is_same< typename Matrix::DeviceType, typename PermutationArray::DeviceType >::value,
                  "The matrix and permutation vector must be stored on the same device." );
   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;
   TNL_ASSERT_EQ( matrix.getRows(), perm.getSize(), "permutation size does not match the matrix size" );

   // FIXME: getConstView does not work
//   const auto matrix_view = matrix.getConstView();
   const auto matrix_view = matrix.getView();
   const auto perm_view = perm.getConstView();

   // create temporary matrix for the permuted data
   Matrix matrixCopy;
   matrixCopy.setLike( matrix );

   // permute the row capacities
   typename Matrix::RowsCapacitiesType capacities( matrix.getRows() );
   auto capacities_view = capacities.getView();

   auto kernel_capacities = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      capacities_view[ i ] = matrix_view.getRowCapacity( perm_view[ i ] );
   };
   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, matrix.getRows(), kernel_capacities );

   matrixCopy.setRowCapacities( capacities );
   auto copy_view = matrixCopy.getView();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      const auto srcRow = matrix_view.getRow( perm_view[ i ] );
      auto destRow = copy_view.getRow( i );
      for( IndexType c = 0; c < srcRow.getSize(); c++ )
         if( srcRow.isBinary() )
            destRow.setElement( c, srcRow.getColumnIndex( c ), true );  // the value does not matter
         else
            destRow.setElement( c, srcRow.getColumnIndex( c ), srcRow.getValue( c ) );
   };
   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, matrix.getRows(), kernel );

   // copy the permuted data back into the matrix
   matrix = matrixCopy;
}

template< typename Matrix,
          typename PermutationArray >
void permuteMatrixColumns( Matrix& matrix, const PermutationArray& iperm )
{
   static_assert( std::is_same< typename Matrix::DeviceType, typename PermutationArray::DeviceType >::value,
                  "The matrix and permutation vector must be stored on the same device." );
   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;

   auto matrix_view = matrix.getView();
   const auto iperm_view = iperm.getConstView();

   auto kernel = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      auto row = matrix_view.getRow( i );
      for( IndexType c = 0; c < row.getSize(); c++ ) {
         const IndexType col = row.getColumnIndex( c );
         if( col == matrix_view.getPaddingIndex() )
            break;
         if( row.isBinary() )
            row.setElement( c, iperm_view[ col ], true );  // the value does not matter
         else
            row.setElement( c, iperm_view[ col ], row.getValue( c ) );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, matrix.getRows(), kernel );
}

} // namespace Matrices
} // namespace TNL
