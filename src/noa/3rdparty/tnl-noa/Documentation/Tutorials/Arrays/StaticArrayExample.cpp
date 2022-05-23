#include <iostream>
#include <TNL/Containers/StaticArray.h>
#include <TNL/File.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   StaticArray< 3, int > a1, a2( 1, 2, 3 ), a3{ 4,3,2 };
   a1 = 0.0;

   std::cout << "a1 = " << a1 << std::endl;
   std::cout << "a2 = " << a2 << std::endl;
   std::cout << "a3 = " << a3 << std::endl;

   File( "static-array-example-file.tnl", std::ios::out ) << a3;
   File( "static-array-example-file.tnl", std::ios::in ) >> a1;

   std::cout << "a1 = " << a1 << std::endl;
   a1.sort();
   std::cout << "Sorted a1 = " << a1 << std::endl;
}

