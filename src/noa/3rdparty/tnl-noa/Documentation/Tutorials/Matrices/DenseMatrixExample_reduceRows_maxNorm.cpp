#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

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
    * Find largest element in each row.
    */
   TNL::Containers::Vector< double, Device > rowMax( matrix.getRows() );

   /***
    * Prepare vector view for lambdas.
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
   matrix.reduceRows( 0, matrix.getRows(), fetch, reduce, keep, std::numeric_limits< double >::lowest() );

   std::cout << "Max. elements in rows are: " << rowMax << std::endl;
   std::cout << "Max. matrix norm is: " << TNL::max( rowMax ) << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Rows reduction on host:" << std::endl;
   reduceRows< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Rows reduction on CUDA device:" << std::endl;
   reduceRows< TNL::Devices::Cuda >();
#endif
}
