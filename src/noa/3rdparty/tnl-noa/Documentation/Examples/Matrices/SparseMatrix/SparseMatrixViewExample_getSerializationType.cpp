#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void getSerializationTypeExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix;
   auto view = matrix.getView();

   std::cout << "Matrix type is: " << view.getSerializationType();
}

int main( int argc, char* argv[] )
{
   std::cout << "Get serialization type on CPU ... " << std::endl;
   getSerializationTypeExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Get serialization type on CUDA GPU ... " << std::endl;
   getSerializationTypeExample< TNL::Devices::Cuda >();
#endif
}
