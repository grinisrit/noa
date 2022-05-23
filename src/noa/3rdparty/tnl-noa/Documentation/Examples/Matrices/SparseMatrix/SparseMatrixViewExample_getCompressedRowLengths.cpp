#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void getCompressedRowLengthsExample()
{
   TNL::Matrices::SparseMatrix< double, Device > triangularMatrix( 5, 5 );
   triangularMatrix.setElements( {
      { 0, 0,  1 },
      { 1, 0,  2 }, { 1, 1,  3 },
      { 2, 0,  4 }, { 2, 1,  5 }, { 2, 2,  6 },
      { 3, 0,  7 }, { 3, 1,  8 }, { 3, 2,  9 }, { 3, 3, 10 },
      { 4, 0, 11 }, { 4, 1, 12 }, { 4, 2, 13 }, { 4, 3, 14 }, { 4, 4, 15 } } );

   std::cout << triangularMatrix << std::endl;

   auto view = triangularMatrix.getView();
   TNL::Containers::Vector< int, Device > rowLengths;
   view.getCompressedRowLengths( rowLengths );

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
