#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create new arrays
    */
   const int size = 15;
   Array< float > a( size ), b( 5 );
   a = 1;
   b = 2;

   /****
    * Create array view for the middle of array a
    */
   auto a_view = a.getView( 5, 10 );

   /****
    * Save array b to file
    */
   b.save( "b.tnl" );

   /****
    * Load data from b to a_view
    */
   a_view.load( "b.tnl" );

   /****
    * Print the results
    */
   std::cout << "a = " << a << std::endl;
   std::cout << "a_view = " << a_view << std::endl;
   std::cout << "b = " << b << std::endl;
}

