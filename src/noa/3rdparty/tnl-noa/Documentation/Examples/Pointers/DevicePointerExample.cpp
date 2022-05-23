#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/DevicePointer.h>

using namespace TNL;

using ArrayCuda = Containers::Array< int, Devices::Cuda >;

struct Tuple
{
   Tuple( ArrayCuda& _a1, ArrayCuda& _a2 ):
   a1( _a1 ), a2( _a2 ){};

   Pointers::DevicePointer< ArrayCuda > a1, a2;
};

#ifdef HAVE_CUDA
__global__ void printTuple( const Tuple t )
{
   printf( "Tuple size is: %d\n", t.a1->getSize() );
   for( int i = 0; i < t.a1->getSize(); i++ )
   {
      printf( "a1[ %d ] = %d \n", i, ( *t.a1 )[ i ] );
      printf( "a2[ %d ] = %d \n", i, ( *t.a2 )[ i ] );
   }
}
#endif

int main( int argc, char* argv[] )
{
   /***
    * Create a tuple of arrays and print them in CUDA kernel
    */
#ifdef HAVE_CUDA
   ArrayCuda a1( 3 ), a2( 3 );
   Tuple t( a1, a2 );
   a1 = 1;
   a2 = 2;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printTuple<<< 1, 1 >>>( t );

   /***
    * Resize the arrays
    */
   a1.setSize( 5 );
   a2.setSize( 5 );
   a1 = 3;
   a2 = 4;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printTuple<<< 1, 1 >>>( t );
#endif
   return EXIT_SUCCESS;

}

