#include <iostream>
#include <TNL/Containers/StaticVector.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   using Vector = StaticVector< 3, double >;

   Vector v1( 1.0 ), v2( 1.0, 2.0, 3.0 ), v3( v1 - v2 / 2.0 );
   Vector v4( 0.0 ), v5( 1.0 );

   v4 += v2;
   v5 *= v3;
   std::cout << "v1 = " << v1 << std::endl;
   std::cout << "v2 = " << v2 << std::endl;
   std::cout << "v3 = " << v3 << std::endl;
   std::cout << "v4 = " << v4 << std::endl;
   std::cout << "v5 = " << v5 << std::endl;
   std::cout << "abs( v3 - 2.0 ) = " << abs( v3 - 2.0 ) << std::endl;
   std::cout << "v2 * v2 = " << v2 * v2 << std::endl;
}

