#include <iostream>
#include <map>
#include <utility>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void setElementsExample()
{
   std::map< std::pair< int, int >, double > map;
   map.insert( std::make_pair( std::make_pair( 0, 0 ),  2.0 ) );
   map.insert( std::make_pair( std::make_pair( 1, 0 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 1, 1 ),  2.0 ) );
   map.insert( std::make_pair( std::make_pair( 1, 2 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 2, 1 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 2, 2 ),  2.0 ) );
   map.insert( std::make_pair( std::make_pair( 2, 3 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 3, 2 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 3, 3 ),  2.0 ) );
   map.insert( std::make_pair( std::make_pair( 3, 4 ), -1.0 ) );
   map.insert( std::make_pair( std::make_pair( 4, 4 ),  2.0 ) );

   TNL::Matrices::SparseMatrix< double, Device > matrix ( 5, 5 );
   matrix.setElements( map );

   std::cout << "General sparse matrix: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ... " << std::endl;
   setElementsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrices on CUDA GPU ... " << std::endl;
   setElementsExample< TNL::Devices::Cuda >();
#endif
}
