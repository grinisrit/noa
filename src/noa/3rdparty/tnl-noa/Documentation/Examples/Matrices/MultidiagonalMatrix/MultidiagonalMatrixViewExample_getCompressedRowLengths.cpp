#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void laplaceOperatorMatrix()
{
   const int gridSize( 4 );
   const int matrixSize = gridSize * gridSize;
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( 
      matrixSize,                     // number of rows
      matrixSize,                     // number of columns
   { - gridSize, -1, 0, 1, gridSize } // diagonals offsets
   );
   matrix.setElements( {
         {  0.0,  0.0, 1.0 },  // set matrix elements corresponding to boundary grid nodes
         {  0.0,  0.0, 1.0 },  // and Dirichlet boundary conditions, i.e. 1 on the main diagonal
         {  0.0,  0.0, 1.0 },  // which is the third one
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 }, // set matrix elements corresponding to inner grid nodes, i.e. 4 on the main diagonal
         { -1.0, -1.0, 4.0, -1.0, -1.0 }, //  (the third one) and -1 to the other sub-diagonals
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         { -1.0, -1.0, 4.0, -1.0, -1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 },
         {  0.0,  0.0, 1.0 }
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
