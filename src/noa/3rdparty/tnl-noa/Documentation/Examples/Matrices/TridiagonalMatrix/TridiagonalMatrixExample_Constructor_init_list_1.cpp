#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void createTridiagonalMatrix()
{
   const int matrixSize = 6;

   /***
    * Setup the following matrix (dots represent zeros):
    * 
    * /  2 -1 .   .  .  . \
    * | -1  2 -1  .  .  . |
    * |  . -1  2 -1  .  . |
    * |  .  . -1  2 -1  . |
    * |  .  .  . -1  2 -1 |
    * \  .  .  .  . -1  2 /
    * 
    */
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix( 
      matrixSize, {
   /***
    * To set the matrix elements we first extend the diagonals to their full
    * lengths even outside the matrix (dots represent zeros and zeros are
    * artificial zeros used for memory alignment):
    * 
    * 0 /  2 -1 .   .  .  . \    -> {  0,  2, -1 }
    *   | -1  2 -1  .  .  . |    -> { -1,  2, -1 }
    *   |  . -1  2 -1  .  . |    -> { -1,  2, -1 }
    *   |  .  . -1  2 -1  . |    -> { -1,  2, -1 }
    *   |  .  .  . -1  2 -1 |    -> { -1,  2, -1 }
    *   \  .  .  .  . -1  2 / 0  -> { -1,  2,  0 }
    * 
    */
      {  0,  2, -1 },
      { -1,  2, -1 },
      { -1,  2, -1 },
      { -1,  2, -1 },
      { -1,  2, -1 },
      { -1,  2,  0 }
      } );
   std::cout << "The matrix reads as: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating tridiagonal matrix on CPU ... " << std::endl;
   createTridiagonalMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating tridiagonal matrix on CUDA GPU ... " << std::endl;
   createTridiagonalMatrix< TNL::Devices::Cuda >();
#endif
}
