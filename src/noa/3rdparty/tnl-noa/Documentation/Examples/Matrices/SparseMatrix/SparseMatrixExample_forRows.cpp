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
    *   |  1  2  1  .  . |
    *   |  .  1  2  1. . |
    *   |  .  .  1  2  1 |
    *   \  .  .  .  .  2 /
    */
   const int size( 5 );
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   MatrixType matrix( { 1, 3, 3, 3, 1 }, size );
   using RowView = typename MatrixType::RowView;

   /***
    * Set the matrix elements.
    */
   auto f = [=] __cuda_callable__ ( RowView& row ) mutable {
      const int rowIdx = row.getRowIndex();
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
   matrix.forAllRows( f );
   std::cout << matrix << std::endl;

   /***
    * Divide each matrix row by a sum of all elements in the row - with use of iterators.
    */
   matrix.forAllRows( [=] __cuda_callable__ ( RowView& row ) mutable {
      double sum( 0.0 );
      for( auto element : row )
         sum += element.value();
      for( auto element: row )
         element.value() /= sum;
   } );
   std::cout << "Divide each matrix row by a sum of all elements in the row ... " << std::endl;
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
