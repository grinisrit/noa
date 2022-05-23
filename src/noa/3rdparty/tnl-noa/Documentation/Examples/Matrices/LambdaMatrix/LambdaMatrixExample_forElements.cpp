#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forElementsExample()
{
   /***
    * Lambda functions defining the matrix.
    */
   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int { return columns; };
   auto matrixElements = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = localIdx;
         value = TNL::max( rowIdx - columnIdx + 1, 0 );
   };

   using MatrixFactory = TNL::Matrices::LambdaMatrixFactory< double, Device, int >;
   auto matrix = MatrixFactory::create( 5, 5, matrixElements, rowLengths );

   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   auto denseView = denseMatrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int columnIdx, double value ) mutable {
      denseView.setElement( rowIdx, columnIdx, value );
   };

   matrix.forElements( 0, matrix.getRows(), f );
   std::cout << "Original lambda matrix:" << std::endl << matrix << std::endl;
   std::cout << "Dense matrix:" << std::endl << denseMatrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Copying matrix on host: " << std::endl;
   forElementsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Copying matrix on CUDA device: " << std::endl;
   forElementsExample< TNL::Devices::Cuda >();
#endif
}
