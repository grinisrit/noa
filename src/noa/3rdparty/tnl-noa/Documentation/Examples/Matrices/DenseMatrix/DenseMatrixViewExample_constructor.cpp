#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void createMatrixView()
{
   TNL::Containers::Vector< double, Device > values {
      1,  2,  3,  4,
      5,  6,  7,  8,
      9, 10, 11, 12 };

   /***
    * Create dense matrix view with row major order
    */
   TNL::Matrices::DenseMatrixView< double, Device, int, TNL::Algorithms::Segments::RowMajorOrder > rowMajorMatrix( 3, 4, values.getView() );
   std::cout << "Row major order matrix:" << std::endl;
   std::cout << rowMajorMatrix << std::endl;

   /***
    * Create dense matrix view with column major order
    */
   TNL::Matrices::DenseMatrixView< double, Device, int, TNL::Algorithms::Segments::RowMajorOrder > columnMajorMatrix( 4, 3, values.getView() );
   std::cout << "Column major order matrix:" << std::endl;
   std::cout << columnMajorMatrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix view on host: " << std::endl;
   createMatrixView< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix view on CUDA device: " << std::endl;
   createMatrixView< TNL::Devices::Cuda >();
#endif
}
