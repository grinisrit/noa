#include <iostream>
#include <iomanip>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void getElements()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix {
      {  1,  0,  0,  0,  0 },
      { -1,  2, -1,  0,  0 },
      {  0, -1,  2, -1,  0 },
      {  0,  0, -1,  2, -1 },
      {  0,  0,  0,  0,  1 } };
   auto matrixView = matrix.getConstView();


   for( int i = 0; i < 5; i++ )
   {
      for( int j = 0; j < 5; j++ )
         std::cout << std::setw( 5 ) << std::ios::right << matrixView.getElement( i, i );
      std::cout << std::endl;
   }
}

int main( int argc, char* argv[] )
{
   std::cout << "Get elements on host:" << std::endl;
   getElements< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Get elements on CUDA device:" << std::endl;
   getElements< TNL::Devices::Cuda >();
#endif
}
