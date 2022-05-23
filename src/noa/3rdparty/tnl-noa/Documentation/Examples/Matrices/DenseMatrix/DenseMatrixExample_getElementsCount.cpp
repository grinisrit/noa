#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getElementsCountExample()
{
   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   };

   std::cout << "Matrix elements count is " << triangularMatrix.getAllocatedElementsCount() << "." << std::endl;
   std::cout << "Non-zero matrix elements count is " << triangularMatrix.getNonzeroElementsCount() << "." << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Computing matrix elements on host: " << std::endl;
   getElementsCountExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Computing matrix elements on CUDA device: " << std::endl;
   getElementsCountExample< TNL::Devices::Cuda >();
#endif
}
