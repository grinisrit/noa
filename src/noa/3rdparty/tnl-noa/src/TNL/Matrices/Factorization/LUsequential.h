// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>

namespace noa::TNL {
namespace Matrices {
namespace Factorization {

template< typename Matrix >
__cuda_callable__
void
LU_sequential_factorize( Matrix& A )
{
   using IndexType = typename Matrix::IndexType;

   TNL_ASSERT_EQ( A.getRows(), A.getColumns(), "LU factorization is possible only for square matrices" );

   const IndexType n = A.getRows();

   for( IndexType k = 0; k < n; k++ ) {
      const auto pivot = A( k, k );
      // Update the (k+1 .. n-1)-th rows
      for( IndexType j = k + 1; j < n; j++ ) {
         const auto factor = A( j, k ) / pivot;
         // Update elements of k-th column below pivot (i.e. elements of L)
         A( j, k ) = factor;
         // Subtract k-th row from j-th
         for( IndexType i = k + 1; i < n; i++ ) {
            A( j, i ) = A( j, i ) - factor * A( k, i );
         }
      }
   }
}

template< typename Matrix, typename Vector >
__cuda_callable__
void
LU_sequential_solve_inplace( const Matrix& A, Vector& x )
{
   using IndexType = typename Matrix::IndexType;
   static_assert( std::is_signed< IndexType >::value, "LU got a matrix with an unsigned index type (2nd for loop won't work)" );

   TNL_ASSERT_EQ( A.getRows(), A.getColumns(), "LU factorization is possible only for square matrices" );

   const IndexType n = A.getRows();

   // Forward substitution
   for( IndexType k = 1; k < n; k++ )
      for( IndexType j = 0; j < k; j++ )
         x[ k ] -= A( k, j ) * x[ j ];

   // Back substitution
   for( IndexType k = n - 1; k >= 0; k-- ) {
      for( IndexType j = n - 1; j > k; j-- )
         x[ k ] -= A( k, j ) * x[ j ];
      x[ k ] /= A( k, k );
   }
}

template< typename Matrix, typename Vector1, typename Vector2 >
__cuda_callable__
void
LU_sequential_solve( const Matrix& A, const Vector1& b, Vector2& x )
{
   // Copy right hand side
   x = b;

   // Solve in-place
   LU_sequential_solve_inplace( A, x );
}

}  // namespace Factorization
}  // namespace Matrices
}  // namespace noa::TNL
