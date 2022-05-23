#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/SharedPointer.h>

using namespace TNL;

using ArrayCuda = Containers::Array< int, Devices::Cuda >;

struct Tuple
{
   Tuple( const int size ):
   a1( size ), a2( size ){};

   void setSize( const int size )
   {
      a1->setSize( size );
      a2->setSize( size );
   }

   Pointers::SharedPointer< ArrayCuda > a1, a2;
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
   Tuple t( 3 );
   *t.a1 = 1;
   *t.a2 = 2;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printTuple<<< 1, 1 >>>( t );

   /***
    * Resize the arrays
    */
   t.setSize( 5 );
   *t.a1 = 3;
   *t.a2 = 4;
   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   printTuple<<< 1, 1 >>>( t );
#endif
   return EXIT_SUCCESS;

}

