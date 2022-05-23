#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void expressions()
{
   using RealType = float;
   using VectorType = Vector< RealType, Device >;
   using ViewType = VectorView< RealType, Device >;
   
   /****
    * Create vectors
    */
   const int size = 11;
   VectorType a_v( size ), b_v( size ), c_v( size );
   ViewType a = a_v.getView();
   ViewType b = b_v.getView();
   ViewType c = c_v.getView();
   a.forAllElements( [] __cuda_callable__ ( int i, RealType& value ) { value = i; } );
   b.forAllElements( [] __cuda_callable__ ( int i, RealType& value ) { value = i - 5.0; } );
   c = -5;

   std::cout << "a = " << a << std::endl;
   std::cout << "b = " << b << std::endl;
   std::cout << "c = " << c << std::endl;
   auto arg_min_a = argMin( a );
   auto arg_max_a = argMax( a );
   auto arg_min_b = argMin( b );
   auto arg_max_b = argMax( b );
   std::cout << "min( a ) = " << arg_min_a.second << " at " << arg_min_a.first << std::endl;
   std::cout << "max( a ) = " << arg_max_a.second << " at " << arg_max_a.first << std::endl;
   std::cout << "min( b ) = " << arg_min_b.second << " at " << arg_min_b.first << std::endl;
   std::cout << "max( b ) = " << arg_max_b.second << " at " << arg_max_b.first << std::endl;
   std::cout << "min( abs( b ) ) = " << min( abs( b ) ) << std::endl;
   std::cout << "sum( b ) = " << sum( b ) << std::endl;
   std::cout << "sum( abs( b ) ) = " << sum( abs( b ) ) << std::endl;
   std::cout << "Scalar product: ( a, b ) =  " << ( a, b ) << std::endl;
   std::cout << "Scalar product: ( a + 3, abs( b ) / 2 ) =  " << ( a + 3, abs( b ) / 2 ) << std::endl;
   if( abs( a  + b ) <=  abs( a ) + abs( b ) )
      std::cout << "abs( a  + b ) <=  abs( a ) + abs( b ) holds" << std::endl;
}

int main( int argc, char* argv[] )
{
   /****
    * Perform test on CPU
    */
   std::cout << "Expressions on CPU ..." << std::endl;
   expressions< Devices::Host >();

   /****
    * Perform test on GPU
    */
#ifdef HAVE_CUDA
   std::cout << std::endl;
   std::cout << "Expressions on GPU ..." << std::endl;
   expressions< Devices::Cuda >();
#endif
}


