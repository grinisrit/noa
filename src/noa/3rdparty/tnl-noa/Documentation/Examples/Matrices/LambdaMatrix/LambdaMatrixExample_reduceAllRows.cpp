#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void reduceAllRows()
{
   /***
    * Lambda functions defining the matrix.
    */
   auto rowLengths = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx ) -> int { return columns; };
   auto matrixElements = [=] __cuda_callable__ ( const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value ) {
         columnIdx = localIdx;
         value = TNL::max( rowIdx - columnIdx + 1, 0 );
   };

   using MatrixFactory = TNL::Matrices::LambdaMatrixFactory< double, Device, int >;
   auto matrix = MatrixFactory::create( 5, 5, matrixElements, rowLengths );

   /***
    * Find largest element in each row.
    */
   TNL::Containers::Vector< double, Device > rowMax( matrix.getRows() );

   /***
    * Prepare vector view and matrix view for lambdas.
    */
   auto rowMaxView = rowMax.getView();

   /***
    * Fetch lambda just returns absolute value of matrix elements.
    */
   auto fetch = [=] __cuda_callable__ ( int rowIdx, int columnIdx, const double& value ) -> double {
      return TNL::abs( value );
   };

   /***
    * Reduce lambda return maximum of given values.
    */
   auto reduce = [=] __cuda_callable__ ( double& a, const double& b ) -> double {
      return TNL::max( a, b );
   };

   /***
    * Keep lambda store the largest value in each row to the vector rowMax.
    */
   auto keep = [=] __cuda_callable__ ( int rowIdx, const double& value ) mutable {
      rowMaxView[ rowIdx ] = value;
   };

   /***
    * Compute the largest values in each row.
    */
   matrix.reduceAllRows( fetch, reduce, keep, std::numeric_limits< double >::lowest() );

   std::cout << "The matrix reads as: " << std::endl << matrix << std::endl;
   std::cout << "Max. elements in rows are: " << rowMax << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "All rows reduction on host:" << std::endl;
   reduceAllRows< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "All rows reduction on CUDA device:" << std::endl;
   reduceAllRows< TNL::Devices::Cuda >();
#endif
}
