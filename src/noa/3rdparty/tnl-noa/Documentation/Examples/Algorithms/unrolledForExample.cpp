#include <iostream>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/unrolledFor.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create two static vectors
    */
   const int Size( 3 );
   StaticVector< Size, double > a, b;
   a = 1.0;
   b = 2.0;
   double sum( 0.0 );

   /****
    * Compute an addition of a vector and a constant number.
    */
   Algorithms::unrolledFor< int, 0, Size >(
      [&]( int i ) {
         a[ i ] = b[ i ] + 3.14;
         sum += a[ i ];
      }
   );
   std::cout << "a = " << a << std::endl;
   std::cout << "sum = " << sum << std::endl;
}
