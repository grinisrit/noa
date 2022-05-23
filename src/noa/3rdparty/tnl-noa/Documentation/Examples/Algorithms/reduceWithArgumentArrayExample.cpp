#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;

template< typename Device >
void reduceArrayExample()
{
   /****
    * Create new arrays
    */
   const int size = 10;
   Containers::Vector< float, Device > a( size );

   /****
    * Initiate the elements of array `a`
    */
   a.forAllElements( [] __cuda_callable__ ( int i, float& value ) { value = 3 - i; } );

   /****
    * Reduce all elements of array `a`
    */
   std::pair< float, int > result_total = Algorithms::reduceWithArgument( TNL::abs( a ), TNL::MaxWithArg{} );

   /****
    * Print the results
    */
   std::cout << " a = " << a << std::endl;
   std::cout << " abs-max of all elements = " << result_total.first << " at position " << result_total.second << std::endl;
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
