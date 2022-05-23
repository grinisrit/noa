#pragma once

template< typename Matrix, typename PermutationVector >
void
getTrivialOrdering( const Matrix& matrix, PermutationVector& perm, PermutationVector& iperm )
{
   using IndexType = typename Matrix::IndexType;

   // allocate permutation vectors
   perm.setSize( matrix.getRows() );
   iperm.setSize( matrix.getRows() );

   const IndexType N = matrix.getRows() / 2;
   for( IndexType i = 0; i < N; i++ ) {
      perm[ 2 * i ] = i;
      perm[ 2 * i + 1 ] = i + N;
      iperm[ i ] = 2 * i;
      iperm[ i + N ] = 2 * i + 1;
   }
}

