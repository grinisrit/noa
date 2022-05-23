#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>


template< typename Device >
void laplaceOperatorMatrix()
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method.
    */
   const int gridSize( 4 );
   const int matrixSize = gridSize * gridSize;
   TNL::Containers::Vector< int, Device > offsets { - gridSize, -1, 0, 1, gridSize };
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( matrixSize, matrixSize, offsets );
   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int i, int j ) mutable {
      const int elementIdx = j * gridSize + i;
      auto row = matrixView.getRow( elementIdx );
      if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
         row.setElement( 2, 1.0 ); // set matrix elements corresponding to boundary grid nodes
                                   // and Dirichlet boundary conditions, i.e. 1 on the main diagonal
                                   // which is the third one
      else
      {
         row.setElement( 0, -1.0 ); // set matrix elements corresponding to inner grid nodes, i.e.
         row.setElement( 1, -1.0 ); // 4 on the main diagonal (the third one) and -1 to the other
         row.setElement( 2,  4.0 ); // sub-diagonals
         row.setElement( 3, -1.0 );
         row.setElement( 4, -1.0 );
      }
   };
   TNL::Algorithms::ParallelFor2D< Device >::exec( 0, 0, gridSize, gridSize, f );

   std::cout << "Laplace operator matrix: " << std::endl << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating Laplace operator matrix on CUDA GPU ... " << std::endl;
   laplaceOperatorMatrix< TNL::Devices::Cuda >();
#endif
}
