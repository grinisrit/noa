#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create new arrays
    */
   const int size = 10;
   Array< float, Devices::Cuda > a( size ), b( size );
   b = 0;

   /****
    * Create an ArrayView and use it for initiation
    */
   auto a_view = a.getView();
   a_view.forAllElements( [] __cuda_callable__ ( int i, float& value ) { value = i; } );

   /****
    * Initiate elements of b with indexes 0-4 using a_view
    */
   b.getView().forElements( 0, 5, [=] __cuda_callable__ ( int i, float& value ) { value = a_view[ i ] + 4.0; } );

   /****
    * Print the results
    */
   std::cout << " a = " << a << std::endl;
   std::cout << " b = " << b << std::endl;
}

