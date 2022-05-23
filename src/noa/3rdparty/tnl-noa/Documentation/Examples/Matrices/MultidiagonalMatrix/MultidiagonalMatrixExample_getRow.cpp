#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   const int matrixSize( 5 );
   auto diagonalsOffsets = { -1, 0, 1 }; // Variadic templates in SharedPointer
                                         // constructor do not recognize initializer
                                         // list so we give it a hint.
   using MatrixType = TNL::Matrices::MultidiagonalMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix(
      matrixSize,  // number of matrix rows
      matrixSize,  // number of matrix columns
      diagonalsOffsets );

   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      //auto row = matrix->getRow( rowIdx );
      // For some reason the previous line of code is not accepted by nvcc 10.1
      // so we replace it with the following two lines.
      auto ref = matrix.modifyData();
      auto row = ref.getRow( rowIdx );

      if( rowIdx > 0 )
         row.setElement( 0, -1.0 );  // elements below the diagonal
      row.setElement( 1, 2.0 );      // elements on the diagonal
      if( rowIdx < matrixSize - 1 )  // elements above the diagonal
         row.setElement( 2, -1.0 );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use MultidiagonalMatrixView. See
    * MultidiagonalMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrix->getRows(), f );
   std::cout << std::endl << *matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   getRowExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   // It seems that nvcc 10.1 does not handle lambda functions properly.
   // It is hard to make nvcc to compile this example and it does not work
   // properly. We will try it with later version of CUDA.
   //std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   //getRowExample< TNL::Devices::Cuda >();
#endif
}
