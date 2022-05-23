#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Pointers/SmartPointersRegister.h>

template< typename Device >
void setElements()
{
   const int matrixSize( 5 );
   auto diagonalsOffsets = { -1, 0, 1 }; // offsets of tridiagonal matrix
   using Matrix = TNL::Matrices::MultidiagonalMatrix< double, Device >;
   TNL::Pointers::SharedPointer< Matrix > matrix( matrixSize, matrixSize, diagonalsOffsets );
   for( int i = 0; i < 5; i++ )
      matrix->setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << *matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i ) mutable {
      if( i > 0 )
         matrix->setElement( i, i - 1, 1.0 );
      matrix->setElement( i, i, -i );
      if( i < matrixSize - 1 )
         matrix->setElement( i, i + 1, 1.0 );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use MultidiagonalMatrixView. See
    * MultidiagonalMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();
   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrixSize, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << *matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   // It seems that nvcc 10.1 does not handle lambda functions properly. 
   // It is hard to make nvcc to compile this example and it does not work
   // properly. We will try it with later version of CUDA.
   //std::cout << "Set elements on CUDA device:" << std::endl;
   //setElements< TNL::Devices::Cuda >();
#endif
}
