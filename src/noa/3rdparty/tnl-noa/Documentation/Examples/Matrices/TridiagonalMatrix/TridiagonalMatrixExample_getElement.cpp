#include <iostream>
#include <iomanip>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void getElements()
{
   const int matrixSize( 5 );
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix (
      matrixSize,   // number of matrix columns
      {             // matrix elements definition
         {  0.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0,  0.0 }
      } );


   for( int i = 0; i < matrixSize; i++ )
   {
      for( int j = 0; j < matrixSize; j++ )
         std::cout << std::setw( 5 ) << matrix.getElement( i, j );
      std::cout << std::endl;
   }
}

int main( int argc, char* argv[] )
{
   std::cout << "Get elements on host:" << std::endl;
   getElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Get elements on CUDA device:" << std::endl;
   getElements< TNL::Devices::Cuda >();
#endif
}
