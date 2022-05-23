#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double scalarProduct( const Vector< double, Device >& u, const Vector< double, Device >& v )
{
   auto u_view = u.getConstView();
   auto v_view = v.getConstView();

   /***
    * Fetch computes product of corresponding elements of both vectors.
    */
   auto fetch = [=] __cuda_callable__ ( int i ) { return u_view[ i ] * v_view[ i ]; };
   auto reduction = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };
   return reduce< Device >( 0, v_view.getSize(), fetch, reduction, 0.0 );
}

int main( int argc, char* argv[] )
{
   /***
    * The first test on CPU ...
    */
   Vector< double, Devices::Host > host_u( 10 ), host_v( 10 );
   host_u = 1.0;
   host_v.forAllElements( [] __cuda_callable__ ( int i, double& value ) { value = 2 * ( i % 2 ) - 1; } );
   std::cout << "host_u = " << host_u << std::endl;
   std::cout << "host_v = " << host_v << std::endl;
   std::cout << "The scalar product ( host_u, host_v ) is " << scalarProduct( host_u, host_v ) << "." << std::endl;
   std::cout << "The scalar product ( host_v, host_v ) is " << scalarProduct( host_v, host_v ) << "." << std::endl;

   /***
    * ... the second test on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 10 ), cuda_v( 10 );
   cuda_u = 1.0;
   cuda_v.forAllElements( [] __cuda_callable__ ( int i, double& value ) { value = 2 * ( i % 2 ) - 1; } );
   std::cout << "cuda_u = " << cuda_u << std::endl;
   std::cout << "cuda_v = " << cuda_v << std::endl;
   std::cout << "The scalar product ( cuda_u, cuda_v ) is " << scalarProduct( cuda_u, cuda_v ) << "." << std::endl;
   std::cout << "The scalar product ( cuda_v, cuda_v ) is " << scalarProduct( cuda_v, cuda_v ) << "." << std::endl;
#endif
   return EXIT_SUCCESS;
}

