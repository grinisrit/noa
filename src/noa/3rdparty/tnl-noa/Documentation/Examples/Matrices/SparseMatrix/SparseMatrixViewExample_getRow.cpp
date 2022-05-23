#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   /***
    * Set the following matrix (dots represent zero matrix elements):
    *
    *   /  2  .  .  .  . \
    *   |  1  2  1  .  . |
    *   |  .  1  2  1. . |
    *   |  .  .  1  2  1 |
    *   \  .  .  .  .  2 /
    */
   const int size = 5;
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 3, 3, 3, 1 }, size );
   auto view = matrix.getView();

   /***
    * Set the matrix elements.
    */
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = view.getRow( rowIdx );
      if( rowIdx == 0 )
         row.setElement( 0, rowIdx, 2.0 );        // diagonal element
      else if( rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );        // diagonal element
      else
      {
         row.setElement( 0, rowIdx - 1, 1.0 );   // elements below the diagonal
         row.setElement( 1, rowIdx, 2.0 );        // diagonal element
         row.setElement( 2, rowIdx + 1, 1.0 );   // elements above the diagonal
      }
   };
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
