#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;

template< typename Device >
void reduceArrayExample()
{
   /****
    * Create new arrays
    */
   const int size = 10;
   Containers::Array< float, Device > a( size );

   /****
    * Initiate the elements of array `a`
    */
   a.forAllElements( [] __cuda_callable__ ( int i, float& value ) { value = i; } );

   /****
    * Sum all elements of array `a`
    */
   float sum_total = Algorithms::reduce( a, TNL::Plus{} );

   /****
    * Sum last 5 elements of array `a`
    */
   float sum_last_five = Algorithms::reduce( a.getConstView( 5, 10 ), TNL::Plus{} );

   /****
    * Print the results
    */
   std::cout << " a = " << a << std::endl;
   std::cout << " sum of all elements = " << sum_total << std::endl;
   std::cout << " sum of last 5 elements = " << sum_last_five << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Running example on the host system: " << std::endl;
   reduceArrayExample< Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Running example on the CUDA device: " << std::endl;
   reduceArrayExample< Devices::Cuda >();
#endif
}
