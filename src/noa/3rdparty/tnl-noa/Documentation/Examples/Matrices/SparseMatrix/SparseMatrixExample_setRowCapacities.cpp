#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void setRowCapacitiesExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( 5, 5 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 1, 2, 3, 4, 5 };
   matrix.setRowCapacities( rowCapacities );
   for( int row = 0; row < 5; row++ )
      for( int column = 0; column <= row; column++ )
         matrix.setElement( row, column, row - column + 1 );

   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   setRowCapacitiesExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   setRowCapacitiesExample< TNL::Devices::Cuda >();
#endif
}
