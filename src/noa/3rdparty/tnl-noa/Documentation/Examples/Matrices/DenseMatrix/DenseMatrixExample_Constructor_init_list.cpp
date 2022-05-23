#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void initializerListExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix {
      {  1,  2,  3,  4,  5,  6 },
      {  7,  8,  9, 10, 11, 12 },
      { 13, 14, 15, 16, 17, 18 }
   };

   std::cout << "General dense matrix: " << std::endl << matrix << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   };

   std::cout << "Triangular dense matrix: " << std::endl << triangularMatrix << std::endl;
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
