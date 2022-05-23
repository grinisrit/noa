#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/UniquePointer.h>

using namespace TNL;

using ArrayCuda = Containers::Array< int, Devices::Cuda >;

#ifdef HAVE_CUDA
__global__ void printArray( const ArrayCuda* ptr )
{
   printf( "Array size is: %d\n", ptr->getSize() );
   for( int i = 0; i < ptr->getSize(); i++ )
      printf( "a[ %d ] = %d \n", i, ( *ptr )[ i ] );
}
#endif

int main( int argc, char* argv[] )
{
   /***
    * Create an array and print its elements in CUDA kernel
    */
#ifdef HAVE_CUDA
   Pointers::UniquePointer< ArrayCuda > array_ptr( 10 );
   array_ptr.modifyData< Devices::Host >() = 1;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printArray<<< 1, 1 >>>( &array_ptr.getData< Devices::Cuda >() );

   /***
    * Resize the array and print it again
    */
   array_ptr->setSize( 5 );
   array_ptr.modifyData< Devices::Host >() = 2;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printArray<<< 1, 1 >>>( &array_ptr.getData< Devices::Cuda >() );
#endif
   return EXIT_SUCCESS;
}

