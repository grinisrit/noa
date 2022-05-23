#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void createMultidiagonalMatrix()
{
   const int matrixSize = 6;

   /***
    * Setup the following matrix (dots represent zeros):
    * 
    * /  4 -1 .  -1  .  . \
    * | -1  4 -1  . -1  . |
    * |  . -1  4 -1  . -1 |
    * | -1  . -1  4 -1  . |
    * |  . -1  . -1  4 -1 |
    * \  .  .  1  . -1  4 /
    * 
    * The diagonals offsets are { -3, -1, 0, 1, 3 }.
    */
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( 
      matrixSize, { -3, -1, 0, 1, 3 }, {
   /***
    * To set the matrix elements we first extend the diagonals to their full
    * lengths even outside the matrix (dots represent zeros and zeros are
    * artificial zeros used for memory alignment):
    * 
    * 0  .  0 /  4 -1 .  -1  .  . \              -> {  0,  0,  4, -1, -1 }
    * .  0  . | -1  4 -1  . -1  . | .            -> {  0, -1,  4, -1, -1 }
    * .  .  0 |  . -1  4 -1  . -1 | .  .         -> {  0, -1,  4, -1, -1 }
    *    .  . | -1  . -1  4 -1  . | 0  .  .      -> { -1, -1,  4, -1,  0 }
    *       . |  . -1  . -1  4 -1 | .  0  .  .   -> { -1, -1,  4, -1,  0 }
    *         \  .  .  1  . -1  4 / 0  .  0  . . -> { -1, -1,  4,  0,  0 }
    * 
    */
      {  0,  0,  4, -1, -1 },
      {  0, -1,  4, -1, -1 },
      {  0, -1,  4, -1, -1 },
      { -1, -1,  4, -1,  0 },
      { -1, -1,  4, -1,  0 },
      { -1, -1,  4,  0,  0 }
      } );
   std::cout << "The matrix reads as: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Create multidiagonal matrix on CPU ... " << std::endl;
   createMultidiagonalMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating multidiagonal matrix on CUDA GPU ... " << std::endl;
   createMultidiagonalMatrix< TNL::Devices::Cuda >();
#endif
}
