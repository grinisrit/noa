#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/scan.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the prefix sum with an array allocated on CPU.
    */
   Array< double, Devices::Host > host_a( 10 );
   host_a = 1.0;
   std::cout << "host_a = " << host_a << std::endl;
   inplaceInclusiveScan( host_a );
   std::cout << "The prefix sum of the host array is " << host_a << "." << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Array< double, Devices::Cuda > cuda_a( 10 );
   cuda_a = 1.0;
   std::cout << "cuda_a = " << cuda_a << std::endl;
   inplaceInclusiveScan( cuda_a );
   std::cout << "The prefix sum of the CUDA array is " << cuda_a << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}
