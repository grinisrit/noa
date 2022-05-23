#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forAllElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx, int columnIdx, int globalIdx, double& value ) {
      if( rowIdx >= columnIdx )
         value = rowIdx + columnIdx;
   };

   matrixView.forAllElements( f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host: " << std::endl;
   forAllElementsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA device: " << std::endl;
   forAllElementsExample< TNL::Devices::Cuda >();
#endif
}
