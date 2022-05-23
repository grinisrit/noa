#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixWriter.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void matrixWriterExample()
{
   using Matrix = TNL::Matrices::SparseMatrix< double, Device >;
   Matrix matrix (
      5, // number of matrix rows
      5, // number of matrix columns
      {  // matrix elements definition
         {  0,  0,  2.0 },
         {  1,  0, -1.0 }, {  1,  1,  2.0 }, {  1,  2, -1.0 },
         {  2,  1, -1.0 }, {  2,  2,  2.0 }, {  2,  3, -1.0 },
         {  3,  2, -1.0 }, {  3,  3,  2.0 }, {  3,  4, -1.0 },
         {  4,  4,  2.0 } } );

   std::cout << "Matrix: " << std::endl << matrix << std::endl;
   std::cout << "Writing matrix in Gnuplot format into the file matrix-writer-example.gplt ...";
   TNL::Matrices::MatrixWriter< Matrix >::writeGnuplot( "matrix-writer-example.gplt", matrix );
   std::cout << " OK " << std::endl;
   std::cout << "Writing matrix pattern in EPS format into the file matrix-writer-example.eps ...";
   TNL::Matrices::MatrixWriter< Matrix >::writeEps( "matrix-writer-example.eps", matrix );
   std::cout << " OK " << std::endl;
   std::cout << "Writing matrix in MTX format into the file matrix-writer-example.mtx ...";
   TNL::Matrices::MatrixWriter< Matrix >::writeMtx( "matrix-writer-example.mtx", matrix );
   std::cout << " OK " << std::endl;
}

template< typename Device >
void matrixReaderExample()
{
   using SparseMatrix = TNL::Matrices::SparseMatrix< double, Device >;
   SparseMatrix sparseMatrix;

   std::cout << "Reading sparse matrix from MTX file matrix-writer-example.mtx ... ";
   TNL::Matrices::MatrixReader< SparseMatrix >::readMtx( "matrix-writer-example.mtx", sparseMatrix );
   std::cout << " OK " << std::endl;
   std::cout << "Imported matrix is: " << std::endl << sparseMatrix << std::endl;

   using DenseMatrix = TNL::Matrices::DenseMatrix< double, Device >;
   DenseMatrix denseMatrix;

   std::cout << "Reading dense matrix from MTX file matrix-writer-example.mtx ... ";
   TNL::Matrices::MatrixReader< DenseMatrix >::readMtx( "matrix-writer-example.mtx", denseMatrix );
   std::cout << " OK " << std::endl;
   std::cout << "Imported matrix is: " << std::endl << denseMatrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ... " << std::endl;
   matrixWriterExample< TNL::Devices::Host >();
   matrixReaderExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << std::endl << std::endl;
   std::cout << "Creating matrices on CUDA GPU ... " << std::endl;
   matrixWriterExample< TNL::Devices::Cuda >();
   matrixReaderExample< TNL::Devices::Cuda >();
#endif
}
