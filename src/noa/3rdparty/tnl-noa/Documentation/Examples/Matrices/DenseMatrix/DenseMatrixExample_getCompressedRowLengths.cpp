#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getCompressedRowLengthsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix {
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 }
   };

   std::cout << triangularMatrix << std::endl;

   TNL::Containers::Vector< int, Device > rowLengths;
   triangularMatrix.getCompressedRowLengths( rowLengths );

   std::cout << "Compressed row lengths are: " << rowLengths << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting compressed row lengths on host: " << std::endl;
   getCompressedRowLengthsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting compressed row lengths on CUDA device: " << std::endl;
   getCompressedRowLengthsExample< TNL::Devices::Cuda >();
#endif
}
