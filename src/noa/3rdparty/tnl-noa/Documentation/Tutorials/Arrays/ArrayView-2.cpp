#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Allocate your own data
    */
   const int size = 5;
   float* a = new float[ size ];

   /****
    * Wrap the data with an array view
    */
   ArrayView< float > a_view( a, size );
   a_view = -5;

   std::cout << " a_view = " << a_view << std::endl;
   for( int i = 0; i < size; i++ )
      std::cout << a[ i ] << " ";

   /****
    * Free the allocated memory
    */
   delete[] a;
}

