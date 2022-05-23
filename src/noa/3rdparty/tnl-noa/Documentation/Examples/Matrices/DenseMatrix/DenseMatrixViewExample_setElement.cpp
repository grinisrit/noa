#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void setElements()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();
   for( int i = 0; i < 5; i++ )
      matrixView.setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i, int j ) mutable {
      matrixView.addElement( i, j, 5.0 );
   };
   TNL::Algorithms::ParallelFor2D< Device >::exec( 0, 0, 5, 5, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Set elements on CUDA device:" << std::endl;
   setElements< TNL::Devices::Cuda >();
#endif
}
