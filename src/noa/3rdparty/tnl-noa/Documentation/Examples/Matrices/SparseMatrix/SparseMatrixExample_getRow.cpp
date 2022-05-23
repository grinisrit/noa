#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   auto rowCapacities = { 1, 1, 1, 1, 1 }; // Variadic templates in SharedPointer
                                           // constructor do not recognize initializer
                                           // list so we give it a hint.
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix( rowCapacities, 5 );

   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrix->getRow( rowIdx );
      row.setElement( 0, rowIdx, 10 * ( rowIdx + 1 ) );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use SparseMatrixView. See
    * SparseMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrix->getRows(), f );
   std::cout << *matrix << std::endl;
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
