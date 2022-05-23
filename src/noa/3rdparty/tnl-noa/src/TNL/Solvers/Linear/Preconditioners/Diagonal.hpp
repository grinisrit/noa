// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Diagonal.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix >
void
Diagonal< Matrix >::update( const MatrixPointer& matrixPointer )
{
   TNL_ASSERT_GT( matrixPointer->getRows(), 0, "empty matrix" );
   TNL_ASSERT_EQ( matrixPointer->getRows(), matrixPointer->getColumns(), "matrix must be square" );

   diagonal.setSize( matrixPointer->getRows() );

   VectorViewType diag_view( diagonal );

   const auto kernel_matrix = matrixPointer->getView();

   // TODO: Rewrite this with SparseMatrix::forAllElements
   auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      diag_view[ i ] = kernel_matrix.getElement( i, i );
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
}

template< typename Matrix >
void
Diagonal< Matrix >::solve( ConstVectorViewType b, VectorViewType x ) const
{
   x = b / diagonal;
}

template< typename Matrix >
void
Diagonal< Matrices::DistributedMatrix< Matrix > >::update( const MatrixPointer& matrixPointer )
{
   diagonal.setSize( matrixPointer->getLocalMatrix().getRows() );

   LocalViewType diag_view( diagonal );
   // FIXME: SparseMatrix::getConstView is broken
   //   const auto matrix_view = matrixPointer->getLocalMatrix().getConstView();
   const auto matrix_view = matrixPointer->getLocalMatrix().getView();

   if( matrixPointer->getRows() == matrixPointer->getColumns() ) {
      // square matrix, assume global column indices
      const auto row_range = matrixPointer->getLocalRowRange();
      auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
      {
         const IndexType gi = row_range.getGlobalIndex( i );
         diag_view[ i ] = matrix_view.getElement( i, gi );
      };
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
   }
   else {
      // non-square matrix, assume ghost indexing
      TNL_ASSERT_LE( matrixPointer->getLocalMatrix().getRows(),
                     matrixPointer->getLocalMatrix().getColumns(),
                     "the local matrix should have more columns than rows" );
      auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
      {
         diag_view[ i ] = matrix_view.getElement( i, i );
      };
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, diagonal.getSize(), kernel );
   }
}

template< typename Matrix >
void
Diagonal< Matrices::DistributedMatrix< Matrix > >::solve( ConstVectorViewType b, VectorViewType x ) const
{
   ConstLocalViewType diag_view( diagonal );
   const auto b_view = b.getConstLocalView();
   auto x_view = x.getLocalView();

   // compute without ghosts (diagonal includes only local rows)
   x_view = b_view / diag_view;

   // synchronize ghosts
   x.startSynchronization();
}

}  // namespace Preconditioners
}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
