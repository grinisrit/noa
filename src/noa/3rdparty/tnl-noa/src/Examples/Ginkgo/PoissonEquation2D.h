#pragma once

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/LambdaMatrix.h>

/**
 * \brief Generates a linear system based on the five-point stencil FDM
 * discretization of the 2D Poisson equation.
 *
 * \param A_local Sparse matrix corresponding to the local block of the calling
 *                MPI rank.
 * \param b_local Vector corresponding to the local block of the right hand side.
 * \param n Number of grid points along the square edge. I.e., the total number
 *          of grid points is `N = n * n`.
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
   if( n * n < num_procs )
      n = std::sqrt( num_procs ) + 1;
   const Index N = n * n;             // global number of rows
   const double h = 1.0 / ( n + 1 );  // mesh cell size

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

         // The left identity block: position i-n
         if( i - n >= 0 )
            nnz++;

         // The left -1: position i-1
         if( i % n )
            nnz++;

         // The diagonal: position i
         nnz++;

         // The right -1: position i+1
         if( ( i + 1 ) % n )
            nnz++;

         // The right identity block: position i+n
         if( i + n < N )
            nnz++;

         // The row index must be converted from global to local
         capacities_view[ i - ilower ] = nnz;
      } );
   A_local.setRowCapacities( capacities );

   /* Now assemble the local matrix. Each row has at most 5 entries. For
    * example, if n=3:
    *   A = [M -I 0; -I M -I; 0 -I M]
    *   M = [4 -1 0; -1 4 -1; 0 -1 4]
    */
   A_local.forAllRows(
      [ = ] __cuda_callable__( typename Matrix::RowView & row ) mutable
      {
         // The row index must be converted from local to global
         const Index i = ilower + row.getRowIndex();
         int nnz = 0;

         // The left identity block: position i-n
         if( i - n >= 0 )
            row.setElement( nnz++, i - n, -1.0 );

         // The left -1: position i-1
         if( i % n )
            row.setElement( nnz++, i - 1, -1.0 );

         // The diagonal: position i
         row.setElement( nnz++, i, 4.0 );

         // The right -1: position i+1
         if( ( i + 1 ) % n )
            row.setElement( nnz++, i + 1, -1.0 );

         // The right identity block: position i+n
         if( i + n < N )
            row.setElement( nnz++, i + n, -1.0 );
      } );

   // Initialize the right hand side vector
   b_local.setSize( local_size );
   b_local.setValue( h * h );
}

// Functor returning the capacity of given row - this must match
// exactly the sparsity pattern of the matrix
// Note that it cannot be actually defined as a lambda function, because nvcc sucks...
template< typename Index >
struct FivePointStencilCapacitiesFunctor
{
   Index n;
   Index N;
   Index ilower;

   FivePointStencilCapacitiesFunctor( Index n, Index N, Index ilower )
   : n( n ), N( N ), ilower( ilower )
   {}

   __cuda_callable__
   int
   operator()( Index local_rows, Index local_columns, Index local_rowIdx )
   {
      // The row index must be converted from local to global
      const Index i = ilower + local_rowIdx;
      int nnz = 0;

      // The left identity block: position i-n
      if( i - n >= 0 )
         nnz++;

      // The left -1: position i-1
      if( i % n )
         nnz++;

      // The diagonal: position i
      nnz++;

      // The right -1: position i+1
      if( ( i + 1 ) % n )
         nnz++;

      // The right identity block: position i+n
      if( i + n < N )
         nnz++;

      return nnz;
   }
};

// Functor for setting the sparse matrix values and column indexes.
// The matrix is the same five-point stencil as above: For example, if n=3:
//   A = [M -I 0; -I M -I; 0 -I M]
//   M = [4 -1 0; -1 4 -1; 0 -1 4]
// Note that it cannot be actually defined as a lambda function, because nvcc sucks...
template< typename Real, typename Index >
struct FivePointStencilElementsFunctor
{
   Index n;
   Index N;
   Index ilower;

   FivePointStencilElementsFunctor( Index n, Index N, Index ilower )
   : n( n ), N( N ), ilower( ilower )
   {}

   __cuda_callable__
   void
   operator()( Index local_rows, Index local_columns, Index local_rowIdx, Index segmentIdx, Index & columnIdx, Real & value )
   {
      // The row index must be converted from local to global
      const Index i = ilower + local_rowIdx;
      int nnz = 0;

      // The left identity block: position i-n
      if( i - n >= 0 )
         if( nnz++ == segmentIdx ) {
            columnIdx = i - n;
            value = -1.0;
            return;
         }

      // The left -1: position i-1
      if( i % n )
         if( nnz++ == segmentIdx ) {
            columnIdx = i - 1;
            value = -1.0;
            return;
         }

      // The diagonal: position i
      if( nnz++ == segmentIdx ) {
         columnIdx = i;
         value = 4.0;
         return;
      }

      // The right -1: position i+1
      if( ( i + 1 ) % n )
         if( nnz++ == segmentIdx ) {
            columnIdx = i + 1;
            value = -1.0;
            return;
         }

      // The right identity block: position i+n
      if( i + n < N )
         if( nnz++ == segmentIdx ) {
            columnIdx = i + n;
            value = -1.0;
            return;
         }
   }
};

template< typename Vector >
auto
getLambdaMatrix( Vector& b_local, typename Vector::IndexType n, int myid = 0, int num_procs = 1 )
{
   using Real = typename Vector::RealType;
   using Index = typename Vector::IndexType;

   // Set the problem sizes
   // Preliminaries: we want at least one rank per row
   if( n * n < num_procs )
      n = std::sqrt( num_procs ) + 1;
   const Index N = n * n;             // global number of rows
   const double h = 1.0 / ( n + 1 );  // mesh cell size

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

   // Instantiate the functors
   FivePointStencilCapacitiesFunctor< Index > capacitiesFunctor( n, N, ilower );
   FivePointStencilElementsFunctor< Real, Index > elementsFunctor( n, N, ilower );

   // Let each rank create its local matrix in the CSR format in TNL
   auto A_local =
      TNL::Matrices::LambdaMatrixFactory< Real, typename Vector::DeviceType, Index >::create( elementsFunctor, capacitiesFunctor );
   A_local.setDimensions( local_size, N );

   // Initialize the right hand side vector
   b_local.setSize( local_size );
   b_local.setValue( h * h );

   return A_local;
}
