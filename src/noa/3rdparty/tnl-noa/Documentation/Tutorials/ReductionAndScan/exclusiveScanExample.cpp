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
   Array< double, Devices::Host > host_input( 10 ), host_output( 10 );
   host_input = 1.0;
   std::cout << "host_input = " << host_input << std::endl;
   exclusiveScan( host_input, host_output );
   std::cout << "host_output " << host_output << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Array< double, Devices::Cuda > cuda_input( 10 ), cuda_output( 10 );
   cuda_input = 1.0;
   std::cout << "cuda_input = " << cuda_input << std::endl;
   exclusiveScan( cuda_input, cuda_output );
   std::cout << "cuda_output " << cuda_output << std::endl;
#endif
   return EXIT_SUCCESS;
}
