#include <iostream>
#include <functional>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getRowExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix {
      { 1, 0, 0, 0, 0 },
      { 1, 2, 0, 0, 0 },
      { 1, 2, 3, 0, 0 },
      { 1, 2, 3, 4, 0 },
      { 1, 2, 3, 4, 5 }
   };

   /***
    * We need a matrix view to pass the matrix to lambda function even on CUDA device.
    */
   const auto matrixView = matrix.getConstView();

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx ) mutable -> double {
      auto row = matrixView.getRow( rowIdx );
      return row.getValue( rowIdx );
   };

   int trace = TNL::Algorithms::reduce< Device >( 0, matrix.getRows(), fetch, std::plus<>{}, 0 );
   std::cout << "Matrix trace is " << trace << "." << std::endl;
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
