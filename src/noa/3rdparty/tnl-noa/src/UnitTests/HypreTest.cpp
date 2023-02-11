#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include "Containers/VectorHelperFunctions.h"

#ifdef HAVE_HYPRE

#include <TNL/Containers/HypreVector.h>
#include <TNL/Containers/HypreParVector.h>
#include <TNL/Matrices/HypreCSRMatrix.h>
#include <TNL/Matrices/HypreParCSRMatrix.h>
#include <TNL/Solvers/Linear/Hypre.h>

#include <TNL/Containers/Partitioner.h>
#include <TNL/Matrices/SparseMatrix.h>

using namespace TNL;
using namespace TNL::Solvers::Linear;

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, Vector )
{
   TNL::Hypre hypre;

   constexpr int size = 10;

   using Vector = Containers::Vector< double, HYPRE_Device >;
   Vector v( size, 2 );

   Containers::HypreVector a;
   a.setSize( size );
   EXPECT_TRUE( a.getData() );
   EXPECT_EQ( a.getSize(), size );
   a.setValue( 2 );
   EXPECT_EQ( a.getConstView(), v );
   EXPECT_EQ( a.getView(), v );

   Containers::HypreVector b;
   b.bind( a );
   EXPECT_EQ( b.getData(), a.getData() );
   EXPECT_EQ( b.getSize(), a.getSize() );

   a.bind( v );
   EXPECT_EQ( a.getData(), v.getData() );
   EXPECT_EQ( a.getSize(), v.getSize() );

   // test move-constructor and move-assignment
   Containers::HypreVector c = std::move( b );
   b = std::move( c );
}

TEST( HypreTest, AssumedPartitionCheck )
{
   // this should always return 1:
   // https://github.com/hypre-space/hypre/blob/master/src/utilities/ap.c
   ASSERT_EQ( HYPRE_AssumedPartitionCheck(), 1 );
}

template< typename DistributedArray >
auto getDistributedArray( MPI_Comm communicator,
                          typename DistributedArray::IndexType globalSize,
                          typename DistributedArray::IndexType ghosts )
{
   DistributedArray array;

   using LocalRangeType = typename DistributedArray::LocalRangeType;
   const LocalRangeType localRange = Containers::Partitioner< typename DistributedArray::IndexType >::splitRange( globalSize, communicator );
   array.setDistribution( localRange, ghosts, globalSize, communicator );

   using Synchronizer = typename Containers::Partitioner< typename DistributedArray::IndexType >::template ArraySynchronizer< typename DistributedArray::DeviceType >;
   array.setSynchronizer( std::make_shared<Synchronizer>( localRange, ghosts / 2, communicator ) );

   return array;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, ParVector )
{
   TNL::Hypre hypre;

   const MPI_Comm communicator = MPI_COMM_WORLD;
   const int globalSize = 97;  // prime number to force non-uniform distribution
   // some arbitrary even value (but must be 0 if not distributed)
   const int ghosts = (TNL::MPI::GetSize(communicator) > 1) ? 4 : 0;

   using DistributedVector = Containers::DistributedVector< HYPRE_Real, HYPRE_Device, HYPRE_BigInt >;
   auto v = getDistributedArray< DistributedVector >( communicator, globalSize, ghosts );
   v.setValue( 2 );

   auto a = getDistributedArray< Containers::HypreParVector >( communicator, globalSize, ghosts );
   EXPECT_EQ( a.getSize(), v.getSize() );
   EXPECT_EQ( a.getLocalRange(), v.getLocalRange() );
   a.setValue( 2 );
   EXPECT_EQ( a.getConstView(), v );
   EXPECT_EQ( a.getView(), v );

   Containers::HypreParVector b;
   b.bind( v );
   EXPECT_EQ( b.getSize(), v.getSize() );
   EXPECT_EQ( b.getLocalView().getData(), v.getLocalView().getData() );
   EXPECT_EQ( b.getLocalView(), v.getLocalView() );

   // test move-constructor and move-assignment
   Containers::HypreParVector c = std::move( b );
   b = std::move( c );
}

/***
 * Creates the following matrix (dots represent zero matrix elements):
 *
 *   /  2.5 -1    .    .    .   \
 *   | -1    2.5 -1    .    .   |
 *   |  .   -1    2.5 -1.   .   |
 *   |  .    .   -1    2.5 -1   |
 *   \  .    .    .   -1    2.5 /
 */
template< typename MatrixType >
MatrixType getGlobalMatrix( int size )
{
   using Vector = Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType, typename MatrixType::IndexType >;
   MatrixType matrix;
   matrix.setDimensions( size, size );
   Vector capacities( size, 3 );
   capacities.setElement( 0, 2 );
   capacities.setElement( size - 1, 2 );
   matrix.setRowCapacities( capacities );

   auto f = [=] __cuda_callable__ ( typename MatrixType::ViewType::RowView& row ) mutable {
      const int rowIdx = row.getRowIndex();
      if( rowIdx == 0 )
      {
         row.setElement( 0, rowIdx,    2.5 );    // diagonal element
         row.setElement( 1, rowIdx+1, -1 );      // element above the diagonal
      }
      else if( rowIdx == size - 1 )
      {
         row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, rowIdx,    2.5 );    // diagonal element
      }
      else
      {
         row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, rowIdx,    2.5 );    // diagonal element
         row.setElement( 2, rowIdx+1, -1.0 );    // element above the diagonal
      }
   };
   matrix.getView().forAllRows( f );

   return matrix;
}

// returns only a local block of the global matrix created by getGlobalMatrix
template< typename MatrixType >
MatrixType getLocalBlock( typename MatrixType::IndexType global_size,
                          Containers::Subrange< typename MatrixType::IndexType > local_row_range )
{
   using Vector = Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType, typename MatrixType::IndexType >;
   MatrixType matrix;
   matrix.setDimensions( local_row_range.getSize(), global_size );
   Vector capacities( local_row_range.getSize(), 3 );
   if( local_row_range.getBegin() == 0 )
      capacities.setElement( 0, 2 );
   if( local_row_range.getEnd() == global_size )
      capacities.setElement( local_row_range.getSize() - 1, 2 );
   matrix.setRowCapacities( capacities );

   auto f = [=] __cuda_callable__ ( typename MatrixType::ViewType::RowView& row ) mutable {
      const int rowIdx = row.getRowIndex();
      const int colIdx = local_row_range.getBegin() + rowIdx;
      if( colIdx == 0 )
      {
         row.setElement( 0, colIdx,    2.5 );    // diagonal element
         row.setElement( 1, colIdx+1, -1 );      // element above the diagonal
      }
      else if( colIdx == global_size - 1 )
      {
         row.setElement( 0, colIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, colIdx,    2.5 );    // diagonal element
      }
      else
      {
         row.setElement( 0, colIdx-1, -1.0 );    // element below the diagonal
         row.setElement( 1, colIdx,    2.5 );    // diagonal element
         row.setElement( 2, colIdx+1, -1.0 );    // element above the diagonal
      }
   };
   matrix.getView().forAllRows( f );

   return matrix;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, CSRMatrix )
{
   TNL::Hypre hypre;

   constexpr int size = 97;

   using MatrixType = Matrices::SparseMatrix< double, HYPRE_Device >;
   MatrixType matrix = getGlobalMatrix< MatrixType >( size );

   Matrices::HypreCSRMatrix a = getGlobalMatrix< Matrices::HypreCSRMatrix >( size );
   EXPECT_GT( a.getValues().getSize(), 0 );
   EXPECT_GT( a.getColumnIndexes().getSize(), 0 );
   EXPECT_GT( a.getRowOffsets().getSize(), 0 );
   EXPECT_EQ( a.getRows(), size );
   EXPECT_EQ( a.getColumns(), size );
   EXPECT_EQ( a.getConstView(), matrix );
   EXPECT_EQ( a.getView(), matrix );

   Matrices::HypreCSRMatrix b;
   b.bind( matrix );
   EXPECT_EQ( b.getRows(), size );
   EXPECT_EQ( b.getColumns(), size );
   EXPECT_EQ( b.getConstView(), matrix );
   EXPECT_EQ( b.getView(), matrix );

   // test move-constructor and move-assignment
   Matrices::HypreCSRMatrix c = std::move( b );
   b = std::move( c );
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, ParCSRMatrix_wrapCSRMatrix )
{
   TNL::Hypre hypre;

   constexpr int size = 97;
   Matrices::HypreCSRMatrix matrix = getGlobalMatrix< Matrices::HypreCSRMatrix >( size );

   auto a = Matrices::HypreParCSRMatrix::wrapCSRMatrix( matrix );
   EXPECT_EQ( a.getRows(), size );
   EXPECT_EQ( a.getColumns(), size );
   EXPECT_EQ( a.getLocalRowRange(), Containers::Subrange< HYPRE_BigInt >( 0, size ) );
   EXPECT_EQ( a.getLocalColumnRange(), Containers::Subrange< HYPRE_BigInt >( 0, size ) );
   EXPECT_EQ( a.getNonzeroElementsCount(), a.getDiagonalBlock().getNonzeroElementsCount() );

   Matrices::HypreCSRMatrix a_diag = a.getDiagonalBlock();
   // the matrices are not elementwise-equal, ParCSR is reordered such that the
   // diagonal elements are first in the rows
//   EXPECT_EQ( a_diag.getValues(), matrix.getValues() );
//   EXPECT_EQ( a_diag.getColumnIndexes(), matrix.getColumnIndexes() );
//   EXPECT_EQ( a_diag.getRowOffsets(), matrix.getRowOffsets() );
   EXPECT_EQ( a_diag.getValues().getSize(), matrix.getValues().getSize() );
   EXPECT_EQ( a_diag.getColumnIndexes().getSize(), matrix.getColumnIndexes().getSize() );
   EXPECT_EQ( a_diag.getRowOffsets().getSize(), matrix.getRowOffsets().getSize() );
   EXPECT_EQ( a_diag.getRows(), size );
   EXPECT_EQ( a_diag.getColumns(), size );
//   EXPECT_EQ( a_diag.getConstView(), matrix );
//   EXPECT_EQ( a_diag.getView(), matrix );

   Matrices::HypreCSRMatrix a_offd = a.getOffdiagonalBlock();
   EXPECT_EQ( a_offd.getValues().getSize(), 0 );
   EXPECT_EQ( a_offd.getColumnIndexes().getSize(), 0 );
   EXPECT_EQ( a_offd.getRowOffsets().getSize(), 0 );
   EXPECT_EQ( a_offd.getRows(), size );
   EXPECT_EQ( a_offd.getColumns(), 0 );

   // test move-constructor and move-assignment
   Matrices::HypreParCSRMatrix c = std::move( a );
   a = std::move( c );
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, ParCSRMatrix_fromMasterRank )
{
   TNL::Hypre hypre;

   const int rank = TNL::MPI::GetRank( MPI_COMM_WORLD );
   const int nproc = TNL::MPI::GetSize( MPI_COMM_WORLD );

   constexpr int global_size = 97;
   Matrices::HypreCSRMatrix matrix = getGlobalMatrix< Matrices::HypreCSRMatrix >( global_size );

   using DistributedVector = Containers::DistributedVector< double, HYPRE_Device >;
   auto x = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );
   Containers::HypreParVector hypre_x;
   hypre_x.bind( x );

   auto a = Matrices::HypreParCSRMatrix::fromMasterRank( matrix, hypre_x, hypre_x );
   EXPECT_EQ( a.getRows(), global_size );
   EXPECT_EQ( a.getColumns(), global_size );
   EXPECT_EQ( a.getLocalRowRange(), x.getLocalRange() );
   EXPECT_EQ( a.getLocalColumnRange(), x.getLocalRange() );

   Matrices::HypreCSRMatrix a_diag = a.getDiagonalBlock();
   const int local_size = x.getLocalRange().getSize();
   EXPECT_EQ( a_diag.getRowOffsets().getSize(), local_size + 1 );
   EXPECT_EQ( a_diag.getRows(), local_size );
   EXPECT_EQ( a_diag.getColumns(), local_size );

   Matrices::HypreCSRMatrix a_offd = a.getOffdiagonalBlock();
   EXPECT_EQ( a_offd.getRowOffsets().getSize(), local_size + 1 );
   EXPECT_EQ( a_offd.getRows(), local_size );
   const int offd_cols = ( nproc == 1 ) ? 0 : ( rank > 0 && rank < nproc - 1 ) ? 2 : 1;
   EXPECT_EQ( a_offd.getColumns(), offd_cols );

   auto col_map_offd = a.getOffdiagonalColumnsMapping();
   EXPECT_EQ( col_map_offd.getSize(), offd_cols );

   // test move-constructor and move-assignment
   Matrices::HypreParCSRMatrix c = std::move( a );
   a = std::move( c );
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST( HypreTest, ParCSRMatrix_fromLocalBlocks )
{
   TNL::Hypre hypre;

   const int rank = TNL::MPI::GetRank( MPI_COMM_WORLD );
   const int nproc = TNL::MPI::GetSize( MPI_COMM_WORLD );

   constexpr int global_size = 97;

   using DistributedVector = Containers::DistributedVector< double, HYPRE_Device >;
   auto x = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );

   Matrices::HypreCSRMatrix matrix = getLocalBlock< Matrices::HypreCSRMatrix >( global_size, x.getLocalRange() );

   auto a = Matrices::HypreParCSRMatrix::fromLocalBlocks( MPI_COMM_WORLD, global_size, global_size, x.getLocalRange(), x.getLocalRange(), matrix );
   EXPECT_EQ( a.getRows(), global_size );
   EXPECT_EQ( a.getColumns(), global_size );
   EXPECT_EQ( a.getLocalRowRange(), x.getLocalRange() );
   EXPECT_EQ( a.getLocalColumnRange(), x.getLocalRange() );

   Matrices::HypreCSRMatrix a_diag = a.getDiagonalBlock();
   const int local_size = x.getLocalRange().getSize();
   EXPECT_EQ( a_diag.getRowOffsets().getSize(), local_size + 1 );
   EXPECT_EQ( a_diag.getRows(), local_size );
   EXPECT_EQ( a_diag.getColumns(), local_size );

   Matrices::HypreCSRMatrix a_offd = a.getOffdiagonalBlock();
   EXPECT_EQ( a_offd.getRowOffsets().getSize(), local_size + 1 );
   EXPECT_EQ( a_offd.getRows(), local_size );
   const int offd_cols = ( nproc == 1 ) ? 0 : ( rank > 0 && rank < nproc - 1 ) ? 2 : 1;
   EXPECT_EQ( a_offd.getColumns(), offd_cols );

   auto col_map_offd = a.getOffdiagonalColumnsMapping();
   EXPECT_EQ( col_map_offd.getSize(), offd_cols );

   Matrices::HypreCSRMatrix a_local = a.getMergedLocalMatrix();
   EXPECT_EQ( a_local.getRows(), matrix.getRows() );
   EXPECT_EQ( a_local.getColumns(), matrix.getColumns() );
   EXPECT_EQ( a_local.getNonzeroElementsCount(), matrix.getNonzeroElementsCount() );
   // TODO: the merged local matrix still has the diagonal element as the first entry per row
   // and we can't use hypre_CSRMatrixReorder on the original block, because it is not square
//   hypre_CSRMatrixReorder( matrix );
//   EXPECT_EQ( a_local.getView(), matrix.getView() );

   // test move-constructor and move-assignment
   Matrices::HypreParCSRMatrix c = std::move( a );
   a = std::move( c );
}


void solve( Matrices::HypreParCSRMatrix& A,
            Containers::HypreParVector& x,
            Containers::HypreParVector& b )
{
   // create the preconditioner
   HypreDiagScale precond;
//   HypreParaSails precond( A.getCommunicator() );
//   HypreEuclid precond( A.getCommunicator() );
//   HypreILU precond;
//   HypreBoomerAMG precond;

   // initialize the Hypre solver
   HyprePCG solver( A.getCommunicator() );
//   solver.setPrintLevel( 1 );
   solver.setPreconditioner( precond );
   solver.setMatrix( A );
   solver.setTol( 1e-9 );
   solver.setResidualConvergenceOptions( -1, 1e-9 );

   // solve the linear system
   solver.solve( b, x );
}

TEST( HypreTest, solve_seq )
{
   TNL::Hypre hypre;

   constexpr int size = 97;

   // create the global matrix
   using MatrixType = Matrices::HypreCSRMatrix;
   auto global_matrix = getGlobalMatrix< MatrixType >( size );

   // set the dofs and right-hand-side vectors
   using Vector = Containers::Vector< double, HYPRE_Device >;
   Vector x( size, 1.0 );
   Vector b( size );
   global_matrix.getView().vectorProduct( x, b );
   x = 0.0;

   // bind parallel Hypre vectors
   Containers::HypreParVector hypre_x;
   Containers::HypreParVector hypre_b;
   hypre_x.bind( {0, size}, 0, size, MPI_COMM_SELF, x.getView() );
   hypre_b.bind( {0, size}, 0, size, MPI_COMM_SELF, b.getView() );

   // convert the matrix to HypreParCSR
   HYPRE_BigInt row_starts[ 2 ];
   row_starts[ 0 ] = 0;
   row_starts[ 1 ] = size;
   auto matrix = Matrices::HypreParCSRMatrix::fromMasterRank( MPI_COMM_SELF, row_starts, row_starts, global_matrix );

   // solve the linear system
   solve( matrix, hypre_x, hypre_b );

   // verify the solution
   expect_near( x, 1, 1e-8 );
}

TEST( HypreTest, solve_distributed_fromMasterRank )
{
   TNL::Hypre hypre;

   constexpr int global_size = 97;

   // create the dofs and right-hand-side vectors
   using DistributedVector = Containers::DistributedVector< double, HYPRE_Device >;
   auto x = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );
   auto b = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );

   // bind parallel Hypre vectors
   Containers::HypreParVector hypre_x;
   Containers::HypreParVector hypre_b;
   hypre_x.bind( x );
   hypre_b.bind( b );

   // create the global matrix
   using MatrixType = Matrices::HypreCSRMatrix;
   auto global_matrix = getGlobalMatrix< MatrixType >( global_size );

   // distribute the matrix to Hypre format (rank 0 -> all ranks)
   auto matrix = Matrices::HypreParCSRMatrix::fromMasterRank( global_matrix, hypre_x, hypre_b );

   // set the right-hand-side
   x.setValue( 1 );
   HYPRE_ParCSRMatrixMatvec( 1.0, matrix, hypre_x, 0.0, hypre_b );
   x.setValue( 0 );

   // solve the linear system
   solve( matrix, hypre_x, hypre_b );

   // verify the solution
   expect_near( x.getLocalView(), 1, 1e-8 );
}

TEST( HypreTest, solve_distributed_fromLocalBlock )
{
   TNL::Hypre hypre;

   constexpr int global_size = 97;

   // create the dofs and right-hand-side vectors
   using DistributedVector = Containers::DistributedVector< double, HYPRE_Device >;
   auto x = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );
   auto b = getDistributedArray< DistributedVector >( MPI_COMM_WORLD, global_size, 2 );

   // bind parallel Hypre vectors
   Containers::HypreParVector hypre_x;
   Containers::HypreParVector hypre_b;
   hypre_x.bind( x );
   hypre_b.bind( b );

   // create the local matrix block
   Matrices::HypreCSRMatrix local_matrix = getLocalBlock< Matrices::HypreCSRMatrix >( global_size, x.getLocalRange() );

   // create the distributed matrix
   auto matrix = Matrices::HypreParCSRMatrix::fromLocalBlocks( MPI_COMM_WORLD, global_size, global_size, x.getLocalRange(), x.getLocalRange(), local_matrix );

   // set the right-hand-side
   x.setValue( 1 );
   HYPRE_ParCSRMatrixMatvec( 1.0, matrix, hypre_x, 0.0, hypre_b );
   x.setValue( 0 );

   // solve the linear system
   solve( matrix, hypre_x, hypre_b );

   // verify the solution
   expect_near( x.getLocalView(), 1, 1e-8 );
}

#endif
#endif

#include "main_mpi.h"
