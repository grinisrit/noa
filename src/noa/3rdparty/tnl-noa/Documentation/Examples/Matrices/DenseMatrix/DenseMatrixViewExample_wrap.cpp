#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixWrapping.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void wrapMatrixView()
{
   const int rows( 3 ), columns( 4 );
   TNL::Containers::Vector< double, Device > valuesVector {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12 };
   double* values = valuesVector.getData();

   /***
    * Wrap the array `values` to dense matrix view
    */
   auto matrix = TNL::Matrices::wrapDenseMatrix< Device >( rows, columns, values );
   std::cout << "Matrix reads as: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Wraping matrix view on host: " << std::endl;
   wrapMatrixView< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Wraping matrix view on CUDA device: " << std::endl;
   wrapMatrixView< TNL::Devices::Cuda >();
#endif
}
