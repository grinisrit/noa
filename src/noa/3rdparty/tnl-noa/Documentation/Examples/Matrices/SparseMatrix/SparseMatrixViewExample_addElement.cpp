#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void addElements()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 5, 5, 5, 5, 5 }, 5 );
   auto view = matrix.getView();
   for( int i = 0; i < 5; i++ )
      view.setElement( i, i, i );

   std::cout << "Initial matrix is: " << std::endl << matrix << std::endl;

   for( int i = 0; i < 5; i++ )
      for( int j = 0; j < 5; j++ )
         view.addElement( i, j, 1.0, 5.0 );

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
