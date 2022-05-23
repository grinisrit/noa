#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void reduceRows()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix {
      {  1,  0,  0,  0,  0 },
      {  1,  2,  0,  0,  0 },
      {  0,  1,  8,  0,  0 },
      {  0,  0,  1,  9,  0 },
      {  0,  0,  0,  0,  1 } };

   /***
    * Allocate input and output vectors for matrix-vector product
    */
   TNL::Containers::Vector< double, Device > x( matrix.getColumns() ),
                                             y( matrix.getRows() );

   /***
    * Fill the input vectors with ones.
    */
   x = 1.0;

   /***
    * Prepare vector view for lambdas.
    */
   auto xView = x.getView();
   auto yView = y.getView();

   /***
    * Fetch lambda just returns product of appropriate matrix elements and the
    * input vector elements.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const double& value ) -> double {
      return xView[ columnIdx ] * value;
   };

   /***
    * Keep lambda store the result of matrix-vector product to output vector y.
    */
   auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
      yView[ rowIdx ] = value;
   };

   /***
    * Compute matrix-vector product.
    */
   matrix.reduceRows( 0, matrix.getRows(), fetch, std::plus<>{}, keep, 0.0 );

   std::cout << "The matrix reads as:" << std::endl << matrix << std::endl;
   std::cout << "The input vector is:" << x << std::endl;
   std::cout << "Result of matrix-vector multiplication is: " << y << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Rows reduction on host:" << std::endl;
   reduceRows< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << std::endl;
   std::cout << "Rows reduction on CUDA device:" << std::endl;
   reduceRows< TNL::Devices::Cuda >();
#endif
}
