#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/SequentialFor.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void printVector()
{
   const int size( 60 );
   TNL::Containers::Vector< float, Device > v( size, 1.0 );
   auto view = v.getView();
   auto print = [=] __cuda_callable__  ( int i ) mutable {
      if( i % 5 == 0 )
         printf( "v[ %d ] = %f \n", i, view[ i ] );  // we use printf because of compatibility with GPU kernels
   };
   std::cout << "Printing vector using parallel for: " << std::endl;
   Algorithms::ParallelFor< Device >::exec( 0, v.getSize(), print );

   std::cout << "Printing vector using sequential for: " << std::endl;
   Algorithms::SequentialFor< Device >::exec( 0, v.getSize(), print );
}

int main( int argc, char* argv[] )
{
   std::cout << "Example on the host:" << std::endl;
   printVector< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Example on CUDA GPU:" << std::endl;
   printVector< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}

