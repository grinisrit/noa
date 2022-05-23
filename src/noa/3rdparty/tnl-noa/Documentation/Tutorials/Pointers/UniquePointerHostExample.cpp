#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Array.h>
#include <TNL/Pointers/UniquePointer.h>

using namespace TNL;

using ArrayHost = Containers::Array< int, Devices::Host >;

int main( int argc, char* argv[] )
{
   /***
    * Make unique pointer on array on CPU and manipulate the
    * array via the pointer.
    */
   Pointers::UniquePointer< ArrayHost > array_ptr( 10 );
   *array_ptr = 1;
   std::cout << "Array size is " << array_ptr->getSize() << std::endl;
   std::cout << "Array = " << *array_ptr << std::endl;
   return EXIT_SUCCESS;
}


