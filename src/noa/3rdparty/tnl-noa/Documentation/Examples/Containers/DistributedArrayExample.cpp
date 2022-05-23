#include <iostream>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedArray.h>
#include <TNL/MPI/ScopedInitializer.h>

using namespace TNL;
using namespace std;

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void distributedArrayExample()
{
   using ArrayType = Containers::DistributedArray< int, Device >;
   using LocalArrayType = Containers::Array< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   using LocalRangeType = typename ArrayType::LocalRangeType;

   const MPI_Comm communicator = MPI_COMM_WORLD;
   //const int rank = TNL::MPI::GetRank(communicator);
   const int nproc = TNL::MPI::GetSize(communicator);

   /***
    * We set size to prime number to force non-uniform distribution of the distributed array.
    */
   const int size = 97;
   const int ghosts = (nproc > 1) ? 4 : 0;

   const LocalRangeType localRange = Containers::Partitioner< IndexType >::splitRange( size, communicator );
   ArrayType a( localRange, ghosts, size, communicator );
   a.forElements( 0, size, [] __cuda_callable__ ( int idx, int& value ) { value = idx; } );
   //LocalArrayType localArray = a;
   //std::cout << a << std::endl;

}

int main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi(argc, argv);

   std::cout << "The first test runs on CPU ..." << std::endl;
   distributedArrayExample< Devices::Host >();
#ifdef HAVE_CUDA
   std::cout << "The second test runs on GPU ..." << std::endl;
   distributedArrayExample< Devices::Cuda >();
#endif
}
