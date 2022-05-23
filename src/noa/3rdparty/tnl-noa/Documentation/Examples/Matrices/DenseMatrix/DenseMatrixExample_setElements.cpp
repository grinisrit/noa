#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void setElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix;
   matrix.setElements( {
      {  1,  2,  3,  4,  5,  6 },
      {  7,  8,  9, 10, 11, 12 },
      { 13, 14, 15, 16, 17, 18 }
   } );

   std::cout << matrix << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix;
   triangularMatrix.setElements( {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   } );

   std::cout << triangularMatrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Setting matrix elements on host: " << std::endl;
   setElementsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Setting matrix elements on CUDA device: " << std::endl;
   setElementsExample< TNL::Devices::Cuda >();
#endif
}