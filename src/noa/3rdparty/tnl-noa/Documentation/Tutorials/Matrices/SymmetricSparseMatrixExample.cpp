#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void symmetricSparseMatrixExample()
{
   TNL::Matrices::SparseMatrix< double, Device, int, TNL::Matrices::SymmetricMatrix > symmetricMatrix (
      5, // number of matrix rows
      5, // number of matrix columns
      {  // matrix elements definition
         {  0,  0, 1.0 },
         {  1,  0, 2.0 }, {  1,  1,  1.0 },
         {  2,  0, 3.0 }, {  2,  2,  1.0 },
         {  3,  0, 4.0 }, {  3,  3,  1.0 },
         {  4,  0, 5.0 }, {  4,  4,  1.0 } } );

   std::cout << "Symmetric sparse matrix: " << std::endl << symmetricMatrix << std::endl;

   TNL::Containers::Vector< double, Device > inVector( 5, 1.0 ), outVector( 5, 0.0 );
   symmetricMatrix.vectorProduct( inVector, outVector );
   std::cout << "Product with vector " << inVector << " is " << outVector << std::endl << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   symmetricSparseMatrixExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   symmetricSparseMatrixExample< TNL::Devices::Cuda >();
#endif
}
