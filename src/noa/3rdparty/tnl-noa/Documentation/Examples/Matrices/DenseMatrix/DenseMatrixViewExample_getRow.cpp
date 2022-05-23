#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getRowExample()
{
   const int size = 5;
   TNL::Matrices::DenseMatrix< double, Device > matrix( size, size );

   /***
    * Create dense matrix view which can be captured by the following lambda
    * function.
    */
   auto matrixView = matrix.getView();

   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      if( rowIdx > 0 )
         row.setValue( rowIdx - 1, -1.0 );
      row.setValue( rowIdx, rowIdx + 1.0 );
      if( rowIdx < size - 1 )
         row.setValue( rowIdx + 1, -1.0 );
   };

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::ParallelFor< Device >::exec( 0, matrix.getRows(), f );
   std::cout << matrix << std::endl;
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
