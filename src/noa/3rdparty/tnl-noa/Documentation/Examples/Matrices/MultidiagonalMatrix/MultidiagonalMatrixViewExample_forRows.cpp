#include <iostream>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   using MatrixType = TNL::Matrices::MultidiagonalMatrix< double, Device >;
   /***
    * Set the following matrix (dots represent zero matrix elements and zeros are
    * padding zeros for memory alignment):
    *
    *    0 /  2  .  .  .  . \  -> { 0, 0, 1 }
    *      | -1  2 -1  .  . |  -> { 0, 2, 1 }
    *      |  . -1  2 -1. . |  -> { 3, 2, 1 }
    *      |  .  . -1  2 -1 |  -> { 3, 2, 1 }
    *      \  .  .  .  .  2 /  -> { 3, 2, 1 }
    *
    * The diagonals offsets are { -1, 0, 1 }.
    */
    const int size = 5;
    MatrixType matrix(
      size,            // number of matrix rows
      size,            // number of matrix columns
      { -1, 0, 1 } ); // matrix diagonals offsets
   auto view = matrix.getView();

   auto f = [=] __cuda_callable__ ( typename MatrixType::RowView& row ) {
      const int& rowIdx = row.getRowIndex();
      if( rowIdx > 0 )
         row.setElement( 0, -1.0 );  // elements below the diagonal
      row.setElement( 1, 2.0 );      // elements on the diagonal
      if( rowIdx < size - 1 )        // elements above the diagonal
         row.setElement( 2, -1.0 );
   };
   view.forAllRows( f );
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host: " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA device: " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
