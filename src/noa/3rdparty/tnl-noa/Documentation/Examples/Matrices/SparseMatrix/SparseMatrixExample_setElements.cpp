#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void setElementsExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix ( 5, 5 ); // matrix dimensions
   matrix.setElements( {                                          // matrix elements definition
      {  0,  0,  2.0 },
      {  1,  0, -1.0 }, {  1,  1,  2.0 }, {  1,  2, -1.0 },
      {  2,  1, -1.0 }, {  2,  2,  2.0 }, {  2,  3, -1.0 },
      {  3,  2, -1.0 }, {  3,  3,  2.0 }, {  3,  4, -1.0 },
      {  4,  4,  2.0 } } );

   std::cout << matrix << std::endl;
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
