#pragma once

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/LambdaMatrix.h>

/**
 * \brief Generates a linear system based on the seven-point stencil FDM
 * discretization of the 3D Poisson equation.
 *
 * \param A_local Sparse matrix corresponding to the local block of the calling
 *                MPI rank.
 * \param b_local Vector corresponding to the local block of the right hand side.
 * \param n Number of grid points along the cube edge. I.e., the total number
 *          of grid points is `N = n * n * n`.
 * \param myid MPI rank ID of the caller.
 * \param num_procs Number of MPI ranks.
 */
template< typename Matrix, typename Vector >
void
generateStencilMatrix( Matrix& A_local, Vector& b_local, typename Matrix::IndexType n, int myid = 0, int num_procs = 1 )
{
   using Index = typename Matrix::IndexType;

   // Set the problem sizes
   // Preliminaries: we want at least one rank per row
   if( n * n * n < num_procs )
      n = std::cbrt( num_procs ) + 1;
   const Index N = n * n * n;          // global number of rows
   const double h = 1.0 / ( n + 1 );   // mesh cell size

   /* Each rank knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of ranks. */
   Index local_size = N / num_procs;
   Index extra = N - local_size * num_procs;

   Index ilower = local_size * myid;
   ilower += TNL::min( myid, extra );

   Index iupper = local_size * ( myid + 1 );
   iupper += TNL::min( myid + 1, extra );
   iupper = iupper - 1;

   // How many rows do I have?
   local_size = iupper - ilower + 1;

   // Let each rank create its local matrix in the CSR format in TNL
   A_local.setDimensions( local_size, N );

   // Allocate row capacities - this must match exactly the sparsity pattern of
   // the matrix
   typename Matrix::RowsCapacitiesType capacities;
   capacities.setSize( local_size );
   auto capacities_view = capacities.getView();
   TNL::Algorithms::ParallelFor< typename Matrix::DeviceType >::exec(
      //
      ilower,
      iupper + 1,
      [ = ] __cuda_callable__( Index i ) mutable
      {
         int nnz = 0;

         // The z-1 neighbor on the grid: position i-n*n
         if( i - n * n >= 0 )
            nnz++;

         // The y-1 neighbor on the grid: position i-n
         if( i - n >= 0 )
            nnz++;

         // The x-1 neighbor on the grid: position i-1
         if( i % n )
            nnz++;

         // The diagonal: position i
         nnz++;

         // The x+1 neighbor on the grid: position i+1
         if( ( i + 1 ) % n )
            nnz++;

         // The y+1 neighbor on the grid: position i+n
         if( i + n < N )
            nnz++;

         // The z+1 neighbor on the grid: position i+n*n
         if( i + n * n < N )
            nnz++;

         // The row index must be converted from global to local
         capacities_view[ i - ilower ] = nnz;
      } );
   A_local.setRowCapacities( capacities );

   // Now assemble the local matrix. Each row has at most 7 entries.
   A_local.forAllRows(
      [ = ] __cuda_callable__( typename Matrix::RowView & row ) mutable
      {
         // The row index must be converted from local to global
         const Index i = ilower + row.getRowIndex();
         int nnz = 0;

         // The z-1 neighbor on the grid: position i-n*n
         if( i - n * n >= 0 )
            row.setElement( nnz++, i - n * n, -1.0 );

         // The y-1 neighbor on the grid: position i-n
         if( i - n >= 0 )
            row.setElement( nnz++, i - n, -1.0 );

         // The x-1 neighbor on the grid: position i-1
         if( i % n )
            row.setElement( nnz++, i - 1, -1.0 );

         // The diagonal: position i
         row.setElement( nnz++, i, 6.0 );

         // The x+1 neighbor on the grid: position i+1
         if( ( i + 1 ) % n )
            row.setElement( nnz++, i + 1, -1.0 );

         // The y+1 neighbor on the grid: position i+n
         if( i + n < N )
            row.setElement( nnz++, i + n, -1.0 );

         // The z+1 neighbor on the grid: position i+n*n
         if( i + n * n < N )
            row.setElement( nnz++, i + n * n, -1.0 );
      } );

   // Initialize the right hand side vector
   b_local.setSize( local_size );
   b_local.setValue( h * h );
}
