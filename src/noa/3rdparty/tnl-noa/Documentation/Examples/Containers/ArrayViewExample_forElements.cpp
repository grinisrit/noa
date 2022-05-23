#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;

template< typename Device >
void forElementsExample()
{
   /****
    * Create new arrays
    */
   const int size = 10;
   Containers::Array< float, Device > a( size ), b( size );
   b = 0;

   /****
    * Create an ArrayView and use it for initiation of elements of array `a`
    */
   auto a_view = a.getView();
   a_view.forAllElements( [] __cuda_callable__ ( int i, float& value ) { value = i; } );

   /****
    * Initiate elements of array `b` with indexes 0-4 using `a_view`
    */
   b.getView().forElements( 0, 5, [=] __cuda_callable__ ( int i, float& value ) { value = a_view[ i ] + 4.0; } );

   /****
    * Print the results
    */
   std::cout << " a = " << a << std::endl;
   std::cout << " b = " << b << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Running example on the host system: " << std::endl;
   forElementsExample< Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Running example on the CUDA device: " << std::endl;
   forElementsExample< Devices::Cuda >();
#endif
}