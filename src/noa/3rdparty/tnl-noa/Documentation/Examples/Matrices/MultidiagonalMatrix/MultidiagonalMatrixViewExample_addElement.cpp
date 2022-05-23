#include <iostream>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void addElements()
{
   const int matrixSize( 5 );
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix(
      matrixSize,     // number of rows
      matrixSize,     // number of columns
      { -1, 0, 1 } ); // diagonals offsets
   auto view = matrix.getView();
   for( int i = 0; i < matrixSize; i++ )
      view.setElement( i, i, i );

   std::cout << "Initial matrix is: " << std::endl << matrix << std::endl;

   for( int i = 0; i < matrixSize; i++ )
   {
      if( i > 0 )
         view.addElement( i, i - 1, 1.0, 5.0 );
      view.addElement( i, i, 1.0, 5.0 );
      if( i < matrixSize - 1 )
         view.addElement( i, i + 1, 1.0, 5.0 );
   }

   std::cout << "Matrix after addition is: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Add elements on host:" << std::endl;
   addElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Add elements on CUDA device:" << std::endl;
   addElements< TNL::Devices::Cuda >();
#endif
}
