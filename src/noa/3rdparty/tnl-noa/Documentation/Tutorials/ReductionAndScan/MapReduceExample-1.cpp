#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double mapReduce( Vector< double, Device >& u )
{
   auto u_view = u.getView();
   auto fetch = [=] __cuda_callable__ ( int i )->double {
      return u_view[ i ] > 0 ? u_view[ i ] : 0.0; };
   auto reduction = [] __cuda_callable__ ( const double& a, const double& b ) { return a + b; };
   return reduce< Device >( 0, u_view.getSize(), fetch, reduction, 0.0 );
}

int main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_u( 10 );
   host_u.forAllElements( [] __cuda_callable__ ( int i, double& value ) { value = sin( ( double ) i ); } );
   double result = mapReduce( host_u );
   std::cout << "host_u = " << host_u << std::endl;
   std::cout << "Sum of the positive numbers is:" << result << std::endl;
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_u( 10 );
   cuda_u = host_u;
   result = mapReduce( cuda_u );
   std::cout << "cuda_u = " << cuda_u << std::endl;
   std::cout << "Sum of the positive numbers is:" << result << std::endl;
#endif
   return EXIT_SUCCESS;
}

