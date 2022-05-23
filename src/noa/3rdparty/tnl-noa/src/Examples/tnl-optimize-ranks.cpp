#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/optimizeRanks.h>

#ifdef HAVE_CUDA
   using DeviceType = TNL::Devices::Cuda;
#else
   using DeviceType = TNL::Devices::Host;
#endif

int main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi(argc, argv);

   const int rank = TNL::MPI::GetRank();
   const int nproc = TNL::MPI::GetSize();

   // TODO: this is only an example
   using Pattern = TNL::Matrices::DenseMatrix< int, TNL::Devices::Sequential, int >;
   Pattern comm_pattern( nproc, nproc );
   comm_pattern.setValue( 0 );
   for( int i = 0; i < nproc; i++ ) {
      // periodic
      //comm_pattern( i, (i + 1 + nproc) % nproc ) = 1;
      //comm_pattern( i, (i - 1 + nproc) % nproc ) = 1;
      // without periodic boundary
      if( i < nproc - 1 )
         comm_pattern( i, i + 1 ) = 1;
      if( i > 0 )
         comm_pattern( i, i - 1 ) = 1;
   }

   if( rank == 0 )
      std::cout << "Communication pattern:\n" << comm_pattern << std::endl;

   const TNL::MPI::Comm perm_comm = TNL::MPI::optimizeRanks< DeviceType >( MPI_COMM_WORLD, comm_pattern );

   std::cout << "rank " << rank << " remapped to " << perm_comm.rank() << std::endl;

   // measure again to verify (up to measurement errors) the optimization
   const auto cost_matrix_perm = TNL::MPI::measureAlltoallCommunicationCost< DeviceType >( perm_comm );

   if( rank == 0 ) {
      using Vector = TNL::Containers::Vector< double, TNL::Devices::Sequential, int >;
      Vector identity( nproc );
      for( int i = 0; i < nproc; i++ )
         identity[ i ] = i;

      std::cout << "cost matrix after permutation:\n" << cost_matrix_perm << std::endl;
      const auto cost = TNL::MPI::getCommunicationCosts( cost_matrix_perm, comm_pattern, identity );
      std::cout << "cost vector: " << cost << " sum " << TNL::sum( cost ) << std::endl;
   }

   return EXIT_SUCCESS;
}
