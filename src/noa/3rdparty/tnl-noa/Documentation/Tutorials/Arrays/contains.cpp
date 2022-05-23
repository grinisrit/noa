#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/contains.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

int main( int argc, char* argv[] )
{
   /****
    * Create new arrays and initiate them
    */
   const int size = 10;
   Array< float, Devices::Cuda > a( size ), b( size );
   a = 0;
   b.forAllElements( [=] __cuda_callable__ ( int i, float& value ) { value = i; } );

   /****
    * Test the values stored in the arrays
    */
   if( contains( a, 0.0 ) )
      std::cout << "a contains 0" << std::endl;

   if( contains( a, 1.0 ) )
      std::cout << "a contains 1" << std::endl;

   if( contains( b, 0.0 ) )
      std::cout << "b contains 0" << std::endl;

   if( contains( b, 1.0 ) )
      std::cout << "b contains 1" << std::endl;

   if( containsOnlyValue( a, 0.0 ) )
      std::cout << "a contains only 0" << std::endl;

   if( containsOnlyValue( a, 1.0 ) )
      std::cout << "a contains only 1" << std::endl;

   if( containsOnlyValue( b, 0.0 ) )
      std::cout << "b contains only 0" << std::endl;

   if( containsOnlyValue( b, 1.0 ) )
      std::cout << "b contains only 1" << std::endl;

   /****
    * Change the first half of b and test it again
    */
   b.forElements( 0, 5, [=] __cuda_callable__ ( int i, float& value ) { value = 0.0; } );
   if( containsOnlyValue( b, 0.0, 0, 5 ) )
      std::cout << "First five elements of b contains only 0" << std::endl;
}

