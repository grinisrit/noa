#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

/****
 * Set all elements of the vector v to the constant c.
 */
template< typename Device >
void initVector( Vector< double, Device >& v,
                 const double& c )
{
   auto view = v.getView();
   auto init = [=] __cuda_callable__ ( int i ) mutable
   {
      view[ i ] = c;
   };
   ParallelFor< Device >::exec( 0, v.getSize(), init );
}

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the vector initiation on CPU.
    */
   Vector< double, Devices::Host > host_v( 10 );
   initVector( host_v, 1.0 );
   std::cout << "host_v = " << host_v << std::endl;

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   initVector( cuda_v, 1.0 );
   std::cout << "cuda_v = " << cuda_v << std::endl;
#endif
   return EXIT_SUCCESS;
}
