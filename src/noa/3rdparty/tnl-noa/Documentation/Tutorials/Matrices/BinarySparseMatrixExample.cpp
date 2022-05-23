#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Devices/Host.h>


template< typename Device >
void binarySparseMatrixExample()
{
   TNL::Matrices::SparseMatrix< bool, Device, int > binaryMatrix (
      5, // number of matrix rows
      5, // number of matrix columns
      {  // matrix elements definition
         {  0,  0, 1.0 }, {  0,  1, 2.0 }, {  0,  2, 3.0 }, {  0,  3, 4.0 }, {  0,  4, 5.0 },
         {  1,  0, 2.0 }, {  1,  1,  1.0 },
         {  2,  0, 3.0 }, {  2,  2,  1.0 },
         {  3,  0, 4.0 }, {  3,  3,  1.0 },
         {  4,  0, 5.0 }, {  4,  4,  1.0 } } );

   std::cout << "Binary sparse matrix: " << std::endl << binaryMatrix << std::endl;

   TNL::Containers::Vector< double, Device > inVector( 5, 1.1 ), outVector( 5, 0.0 );
   binaryMatrix.vectorProduct( inVector, outVector );
   std::cout << "Product with vector " << inVector << " is " << outVector << std::endl << std::endl;

   TNL::Matrices::SparseMatrix< bool, Device, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault, double > binaryMatrix2;
   binaryMatrix2 = binaryMatrix;
   binaryMatrix2.vectorProduct( inVector, outVector );
   std::cout << "Product with vector in double precision " << inVector << " is " << outVector << std::endl << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   binarySparseMatrixExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   binarySparseMatrixExample< TNL::Devices::Cuda >();
#endif
}
