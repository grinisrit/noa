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

   /***
    * Prepare second array holding elements positions.
    */
   ArrayT index( size );
   index.forAllElements( [] __cuda_callable__ ( int idx, int& value  ) { value = idx; } );
   std::cout << "Random array:     " << array << std::endl;
   std::cout << "Index array:      " << index << std::endl;

   /***
    * Sort the array `array` and apply the same permutation on the array `identity`.
    */
   auto array_view = array.getView();
   auto index_view = index.getView();
   sort< typename ArrayT::DeviceType,                             // device on which the sorting will be performed
         typename ArrayT::IndexType >(                            // type used for indexing
         0, size,                                                 // range of indexes
         [=] __cuda_callable__ ( int i, int j ) -> bool {         // comparison lambda function
            return array_view[ i ] < array_view[ j ]; },
         [=] __cuda_callable__ ( int i, int j ) mutable {         // lambda function for swapping of elements
            TNL::swap( array_view[ i ], array_view[ j ] );
            TNL::swap( index_view[ i ], index_view[ j ] ); } );
   std::cout << "Sorted array:      " << array << std::endl;
   std::cout << "Index:             " << index << std::endl;
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
