#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/sort.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename ArrayT >
void sort( ArrayT& array )
{
   const int size = 10;

  /****
   * Fill the array with random integers.
   */
   Array< int > aux_array( size );
   srand( size + 2021 );
   aux_array.forAllElements( [=] __cuda_callable__ ( int i, int& value ) { value = std::rand() % (2*size); } );
   array = aux_array;

   std::cout << "Random array: " << array << std::endl;

   /****
    * Sort the array in ascending order.
    */
   sort( array, [] __cuda_callable__ ( int a, int b ) { return a < b; } );
   std::cout << "Array sorted in ascending order:" << array << std::endl;

   /***
    * Sort the array in descending order.
    */
   sort( array, [] __cuda_callable__ ( int a, int b ) { return a > b; } );
   std::cout << "Array sorted in descending order:" << array << std::endl;
}

int main( int argc, char* argv[] )
{
   /***
    * Firstly, test the sorting on CPU.
    */
   std::cout << "Sorting on CPU ... " << std::endl;
   Array< int, Devices::Host > host_array;
   sort( host_array );

#ifdef HAVE_CUDA
   /***
    * And then also on GPU.
    */
   std::cout << "Sorting on GPU ... " << std::endl;
   Array< int, Devices::Cuda > cuda_array;
   sort( cuda_array );
#endif
   return EXIT_SUCCESS;
}
