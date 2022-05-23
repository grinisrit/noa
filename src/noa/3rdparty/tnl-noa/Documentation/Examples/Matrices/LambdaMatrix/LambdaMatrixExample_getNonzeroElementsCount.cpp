#include <iostream>
#include <TNL/Matrices/LambdaMatrix.h>

int main( int argc, char* argv[] )
{
   /***
    * Lambda functions defining the matrix.
    */
   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int { return columns; };
   auto matrixElements = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = localIdx;
         value = TNL::max( rowIdx - columnIdx + 1, 0 );
   };

   const int size = 5;
   auto matrix = TNL::Matrices::LambdaMatrixFactory< double, TNL::Devices::Host, int >::create( size, size, matrixElements, rowLengths );

   std::cout << "Matrix looks as:" << std::endl << matrix << std::endl;
   std::cout << "Non-zero elements count is: " << matrix.getNonzeroElementsCount() << std::endl;
}
