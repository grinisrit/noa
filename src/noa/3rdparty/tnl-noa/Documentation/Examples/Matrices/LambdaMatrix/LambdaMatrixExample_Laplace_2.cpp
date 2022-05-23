#include <iostream>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void laplaceOperatorMatrix()
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method.
    */
   const int gridSize( 4 );
   const int matrixSize = gridSize * gridSize;
   TNL::Containers::Vector< int, Device > columnOffsets{ 0, -1, 0, 1, 0 };  // helper vector for getting matrix elements column indexes
   columnOffsets.setElement( 0, -gridSize );
   columnOffsets.setElement( 4, gridSize );
   auto columnOffsetsView = columnOffsets.getView();
   TNL::Containers::Vector< double, Device > values{ 1, 1, -4, 1, 1 };      // helper vector for getting matrix elements values
   auto valuesView = values.getView();

   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int
   {
      const int gridRow = rowIdx / gridSize;                  // coordinates in the numerical grid
      const int gridColumn = rowIdx % gridSize;

      if( gridRow == 0 || gridRow == gridSize - 1 ||          // boundary grid node
          gridColumn == 0 || gridColumn == gridSize - 1 )
          return 1;
      return 5;
   };

   auto matrixElements = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value) {
      const int gridRow = rowIdx / gridSize;                  // coordinates in the numerical grid
      const int gridColumn = rowIdx % gridSize;
      if( gridRow == 0 || gridRow == gridSize - 1 ||          // boundary grid node
          gridColumn == 0 || gridColumn == gridSize - 1 )
         {
            columnIdx = rowIdx;                               // diagonal element ....
            value = 1.0;                                      // ... is set to 1
         }
         else                                                 // interior grid node
         {
            columnIdx = rowIdx + columnOffsetsView[ localIdx ];
            value = valuesView[ localIdx ];
         }
   };
   auto matrix = TNL::Matrices::LambdaMatrixFactory< double, Device, int >::create(
      matrixSize, matrixSize, matrixElements, rowLengths );
   std::cout << "Laplace operator matrix: " << std::endl << matrix << std::endl;
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
