#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void reduceAllRows()
{
   /***
    * Set the following matrix (dots represent zero matrix elements and zeros are
    * padding zeros for memory alignment):
    *
    * 0  0 / 1  .  .  .  . \  -> { 0, 0, 1 }
    *    0 | 2  1  .  .  . |  -> { 0, 2, 1 }
    *      | 3  2  1  .  . |  -> { 3, 2, 1 }
    *      | .  3  2  1  . |  -> { 3, 2, 1 }
    *      \ .  .  3  2  1 /  -> { 3, 2, 1 }
    *
    * The diagonals offsets are { -2, -1, 0 }.
    */
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix (
      5,              // number of matrix columns
      { -2, -1, 0 },  // diagonals offsets
      { { 0, 0, 1 },  // matrix elements
        { 0, 2, 1 },
        { 3, 2, 1 },
        { 3, 2, 1 },
        { 3, 2, 1 } } );

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
   matrix.reduceAllRows( fetch, reduce, keep, std::numeric_limits< double >::lowest() );

   std::cout << "The matrix reads as: " << std::endl << matrix << std::endl;
   std::cout << "Max. elements in rows are: " << rowMax << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Rows reduction on host:" << std::endl;
   reduceAllRows< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Rows reduction on CUDA device:" << std::endl;
   reduceAllRows< TNL::Devices::Cuda >();
#endif
}
