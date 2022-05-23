#include <iostream>
#include <functional>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getRowExample()
{
   const int matrixSize = 5;
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, Device >;
   MatrixType matrix (
      matrixSize,           // number of matrix columns
      {  { 0.0, 2.0, 1.0 }, // matrix elements
         { 0.0, 2.0, 1.0 },
         { 3.0, 2.0, 1.0 },
         { 3.0, 2.0, 1.0 },
         { 0.0, 2.0, 1.0 } } );
   auto view = matrix.getView();

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx ) mutable -> double {
      auto row = view.getRow( rowIdx );
      return row.getValue( 2 ); // get value from subdiagonal with index 2, i.e. the main diagonal
   };

   /***
    * Compute the matrix trace.
    */
   int trace = TNL::Algorithms::reduce< Device >( 0, view.getRows(), fetch, std::plus<>{}, 0 );
   std::cout << "Matrix reads as: " << std::endl << matrix << std::endl;
   std::cout << "Matrix trace is: " << trace << "." << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   getRowExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   getRowExample< TNL::Devices::Cuda >();
#endif
}
