#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void reduceRows()
{
   /***
    * Set the following matrix (dots represent zero matrix elements and zeros are
    * padding zeros for memory alignment):
    *
    *  0 / 1  3  .  .  . \   -> { 0, 1, 3 }
    *    | 2  1  3  .  . |   -> { 2, 1, 3 }
    *    | .  2  1  3  . |   -> { 2, 1, 3 }
    *    | .  .  2  1  3 |   -> { 2, 1, 3 }
    *    \ .  .  .  2  1 / 0 -> { 2, 1, 0 }
    *
    */
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix (
      5,              // number of matrix columns
      { { 0, 1, 3 },  // matrix elements
        { 2, 1, 3 },
        { 2, 1, 3 },
        { 2, 1, 3 },
        { 2, 1, 3 } } );
   auto view = matrix.getView();

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
   view.reduceAllRows( fetch, reduce, keep, std::numeric_limits< double >::lowest() );

   std::cout << "The matrix reads as: " << std::endl << matrix << std::endl;
   std::cout << "Max. elements in rows are: " << rowMax << std::endl;
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
