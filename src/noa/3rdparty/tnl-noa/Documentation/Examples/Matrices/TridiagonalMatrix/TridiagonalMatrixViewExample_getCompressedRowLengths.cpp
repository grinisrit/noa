#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void laplaceOperatorMatrix()
{
   const int gridSize( 6 );
   const int matrixSize = gridSize;
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix( 
      matrixSize, // number of rows
      matrixSize  // number of columns
   );
   matrix.setElements( {
         {  0.0, 1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         {  0.0, 1.0 }
      } );
   auto view = matrix.getView();

   TNL::Containers::Vector< int, Device > rowLengths;
   view.getCompressedRowLengths( rowLengths );
   std::cout << "Laplace operator matrix: " << std::endl << matrix << std::endl;
   std::cout << "Compressed row lengths: " << rowLengths << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating Laplace operator matrix on CUDA GPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Cuda >();
#endif
}
