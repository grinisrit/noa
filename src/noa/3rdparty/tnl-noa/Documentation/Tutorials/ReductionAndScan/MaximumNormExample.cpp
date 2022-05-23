#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double maximumNorm( const Vector< double, Device >& v )
{
   auto view = v.getConstView();
   auto fetch = [=] __cuda_callable__ ( int i ) { return abs( view[ i ] ); };
   auto reduction = [] __cuda_callable__ ( const double& a, const double& b ) { return max( a, b ); };
   return reduce< Device >( 0, view.getSize(), fetch, reduction, 0.0 );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_v( 10 );
   host_v.forAllElements( [] __cuda_callable__ ( int i, double& value ) { value = i - 7; } );
   std::cout << "host_v = " << host_v << std::endl;
   std::cout << "The maximum norm of the host vector elements is " << maximumNorm( host_v ) << "." << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( 10 );
   cuda_v.forAllElements( [] __cuda_callable__ ( int i, double& value ) { value = i - 7; } );
   std::cout << "cuda_v = " << cuda_v << std::endl;
   std::cout << "The maximum norm of the CUDA vector elements is " << maximumNorm( cuda_v ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

