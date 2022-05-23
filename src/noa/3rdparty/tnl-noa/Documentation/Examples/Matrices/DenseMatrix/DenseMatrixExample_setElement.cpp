#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Pointers/SmartPointersRegister.h>

template< typename Device >
void setElements()
{
   TNL::Pointers::SharedPointer< TNL::Matrices::DenseMatrix< double, Device > > matrix( 5, 5 );
   for( int i = 0; i < 5; i++ )
      matrix->setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << *matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i, int j ) mutable {
      matrix->addElement( i, j, 5.0 );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use DenseMatrixView. See
    * DenseMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();
   TNL::Algorithms::ParallelFor2D< Device >::exec( 0, 0, 5, 5, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << *matrix << std::endl;
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
