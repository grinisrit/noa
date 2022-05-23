#include <iostream>
#include <TNL/Matrices/LambdaMatrix.h>

int main( int argc, char* argv[] )
{
   /***
    * Lambda functions defining the matrix.
    */
   auto compressedRowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int { return 1; };
   auto matrixElements1 = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };
   auto matrixElements2 = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = rowIdx;
         value =  rowIdx;
   };

   const int size = 5;

   /***
    * Matrix construction with explicit type definition.
    */
   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< double, TNL::Devices::Host, int >::create( matrixElements1, compressedRowLengths ) );
   MatrixType m1( size, size, matrixElements1, compressedRowLengths );

   /***
    * Matrix construction using 'auto'.
    */
   auto m2 = TNL::Matrices::LambdaMatrixFactory< double, TNL::Devices::Host, int >::create( matrixElements2, compressedRowLengths );
   m2.setDimensions( size, size );

   std::cout << "The first lambda matrix: " << std::endl << m1 << std::endl;
   std::cout << "The second lambda matrix: " << std::endl << m2 << std::endl;
}
