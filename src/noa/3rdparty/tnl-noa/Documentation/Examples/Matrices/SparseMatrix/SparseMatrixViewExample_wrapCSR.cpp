#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixWrapping.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void wrapMatrixView()
{
   /***
    * Encode the following matrix to CSR format...
    *
    * /  1  2  0  0 \.
    * |  0  6  0  0 |
    * |  9  0  0  0 |
    * \  0  0 15 16 /
    */
   const int rows( 4 ), columns( 4 );
   TNL::Containers::Vector< double, Device > valuesVector     { 1, 2, 6, 9, 15, 16 };
   TNL::Containers::Vector< int, Device > columnIndexesVector { 0, 1, 1, 0,  2,  3 };
   TNL::Containers::Vector< int, Device > rowPointersVector   { 0, 2, 3, 4, 6 };

   double* values = valuesVector.getData();
   int* columnIndexes = columnIndexesVector.getData();
   int* rowPointers = rowPointersVector.getData();

   /***
    * Wrap the arrays `rowPointers, `values` and `columnIndexes` to sparse matrix view
    */
   auto matrix = TNL::Matrices::wrapCSRMatrix< Device >( rows, columns, rowPointers, values, columnIndexes );

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
