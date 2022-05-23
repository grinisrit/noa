#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void initializerListExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix {
      {  1,  2,  3,  4,  5 }, // row capacities
      6 };                    // number of matrix columns

   for( int row = 0; row < matrix.getRows(); row++ )
      for( int column = 0; column <= row; column++ )
         matrix.setElement( row, column, row - column + 1 );
   std::cout << "General sparse matrix: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ... " << std::endl;
   initializerListExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrices on CUDA GPU ... " << std::endl;
   initializerListExample< TNL::Devices::Cuda >();
#endif
}
