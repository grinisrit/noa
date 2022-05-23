#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void arrayViewExample()
{
   const int size = 10;
   using ArrayType = Containers::Array< int, Device >;
   using IndexType = typename ArrayType::IndexType;
   using ViewType = Containers::ArrayView< int, Device >;
   ArrayType a1( size ), a2( size );
   ViewType a1_view = a1.getView();
   ViewType a2_view = a2.getView();

   /***
    * You may initiate the array view using setElement
    */
   for( int i = 0; i < size; i++ )
      a1_view.setElement( i, i );

   /***
    * You may also assign value to all array view elements ...
    */
   a2_view = 0;

   /***
    * More efficient way of array view elements manipulation is with the lambda functions
    */
   ArrayType a3( size );
   ViewType a3_view = a3.getView();
   auto f1 = [] __cuda_callable__ ( IndexType i, int& value ) { value = 2 * i; };
   a3_view.forAllElements( f1 );

   for( int i = 0; i < size; i++ )
      if( a3_view.getElement( i ) != 2 * i )
         std::cerr << "Something is wrong!!!" << std::endl;

   /***
    * You may swap array view data with the swap method.
    */
   a1_view.swap( a3_view );

   /***
    * You may save it to file and load again
    */
   File( "a1_view.tnl", std::ios_base::out ) << a1_view;
   File( "a1_view.tnl", std::ios_base::in ) >> a2_view;

   std::remove( "a1_view.tnl" );

   if( a2_view != a1_view )
      std::cerr << "Something is wrong!!!" << std::endl;

   std::cout << "a2_view = " << a2_view << std::endl;
}

int main()
{
   std::cout << "The first test runs on CPU ..." << std::endl;
   arrayViewExample< Devices::Host >();
#ifdef HAVE_CUDA
   std::cout << "The second test runs on GPU ..." << std::endl;
   arrayViewExample< Devices::Cuda >();
#endif
}
