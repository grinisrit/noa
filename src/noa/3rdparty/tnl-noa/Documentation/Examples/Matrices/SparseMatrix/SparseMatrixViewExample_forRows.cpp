#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   /***
    * Set the following matrix (dots represent zero matrix elements):
    *
    *   /  2  .  .  .  . \
    *   | -1  2 -1  .  . |
    *   |  . -1  2 -1. . |
    *   |  .  . -1  2 -1 |
    *   \  .  .  .  .  2 /
    */
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   const int size( 5 );
   MatrixType matrix( { 1, 3, 3, 3, 1 }, size );
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( typename MatrixType::RowView& row ) mutable {
      const int rowIdx = row.getRowIndex();
      if( rowIdx == 0 )
         row.setElement( 0, rowIdx, 2.0 );        // diagonal element
      else if( rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );        // diagonal element
      else
      {
         row.setElement( 0, rowIdx - 1, -1.0 );   // elements below the diagonal
         row.setElement( 1, rowIdx, 2.0 );        // diagonal element
         row.setElement( 2, rowIdx + 1, -1.0 );   // elements above the diagonal
      }
   };

   /***
    * Set the matrix elements.
    */
   matrix.forAllRows( f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
