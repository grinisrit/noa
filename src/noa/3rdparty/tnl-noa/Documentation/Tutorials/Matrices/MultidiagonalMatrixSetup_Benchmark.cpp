#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Timer.h>

const int testsCount = 5;

template< typename Device >
TNL::Containers::Vector< int, Device > getOffsets( const int gridSize )
{
   TNL::Containers::Vector< int, Device > offsets( 5 );
   offsets.setElement( 0, -gridSize );
   offsets.setElement( 1, -1 );
   offsets.setElement( 2, 0 );
   offsets.setElement( 3, 1 );
   offsets.setElement( 4, gridSize );
   return offsets;
}

template< typename Matrix >
void setElement_on_host( const int gridSize, Matrix& matrix )
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method by means setElement method called
    * from the host system.
    */
   const int matrixSize = gridSize * gridSize;
   matrix.setDimensions( matrixSize, matrixSize, getOffsets< typename Matrix::DeviceType >( gridSize ) );

   for( int j = 0; j < gridSize; j++ )
      for( int i = 0; i < gridSize; i++ )
      {
         const int rowIdx = j * gridSize + i;
         if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
            matrix.setElement( rowIdx, rowIdx,  1.0 );
         else
         {
            matrix.setElement( rowIdx, rowIdx - gridSize,  1.0 );
            matrix.setElement( rowIdx, rowIdx - 1,  1.0 );
            matrix.setElement( rowIdx, rowIdx,  -4.0 );
            matrix.setElement( rowIdx, rowIdx + 1,  1.0 );
            matrix.setElement( rowIdx, rowIdx + gridSize,  1.0 );
         }
      }
}

template< typename Matrix >
void setElement_on_host_and_transfer( const int gridSize, Matrix& matrix )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using HostMatrix = TNL::Matrices::MultidiagonalMatrix< RealType, TNL::Devices::Host, IndexType >;
   const int matrixSize = gridSize * gridSize;
   HostMatrix hostMatrix( matrixSize, matrixSize, getOffsets< typename Matrix::DeviceType >( gridSize ) );

   for( int j = 0; j < gridSize; j++ )
      for( int i = 0; i < gridSize; i++ )
      {
         const int rowIdx = j * gridSize + i;
         if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
            hostMatrix.setElement( rowIdx, rowIdx,  1.0 );
         else
         {
            hostMatrix.setElement( rowIdx, rowIdx - gridSize,  1.0 );
            hostMatrix.setElement( rowIdx, rowIdx - 1,  1.0 );
            hostMatrix.setElement( rowIdx, rowIdx,  -4.0 );
            hostMatrix.setElement( rowIdx, rowIdx + 1,  1.0 );
            hostMatrix.setElement( rowIdx, rowIdx + gridSize,  1.0 );
         }
      }
   matrix = hostMatrix;
}


template< typename Matrix >
void setElement_on_device( const int gridSize, Matrix& matrix )
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method by means of setElement method called
    * from the native device.
    */
   const int matrixSize = gridSize * gridSize;
   matrix.setDimensions( matrixSize, matrixSize, getOffsets< typename Matrix::DeviceType >( gridSize ) );

   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int i, int j ) mutable {
      const int rowIdx = j * gridSize + i;
      if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
         matrixView.setElement( rowIdx, rowIdx,  1.0 );
      else
      {
         matrixView.setElement( rowIdx, rowIdx - gridSize,  1.0 );
         matrixView.setElement( rowIdx, rowIdx - 1,  1.0 );
         matrixView.setElement( rowIdx, rowIdx,  -4.0 );
         matrixView.setElement( rowIdx, rowIdx + 1,  1.0 );
         matrixView.setElement( rowIdx, rowIdx + gridSize,  1.0 );
      }
   };
   TNL::Algorithms::ParallelFor2D< typename Matrix::DeviceType >::exec( 0, 0, gridSize, gridSize, f );
}

template< typename Matrix >
void getRow( const int gridSize, Matrix& matrix )
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method by means of getRow method.
    */
   const int matrixSize = gridSize * gridSize;
   matrix.setDimensions( matrixSize, matrixSize, getOffsets< typename Matrix::DeviceType >( gridSize ) );

   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      const int i = rowIdx % gridSize;
      const int j = rowIdx / gridSize;
      auto row = matrixView.getRow( rowIdx );
      if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
         row.setElement( 2, 1.0 );
      else
      {
         row.setElement( 0, 1.0 );
         row.setElement( 1, 1.0 );
         row.setElement( 2, -4.0 );
         row.setElement( 3, 1.0 );
         row.setElement( 4, 1.0 );
      }
   };
   TNL::Algorithms::ParallelFor< typename Matrix::DeviceType >::exec( 0, matrixSize, f );
}

template< typename Matrix >
void forElements( const int gridSize, Matrix& matrix )
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method by means of forElements method.
    */

   const int matrixSize = gridSize * gridSize;
   matrix.setDimensions( matrixSize, matrixSize, getOffsets< typename Matrix::DeviceType >( gridSize ) );

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int columnIdx, float& value ) mutable {
      const int i = rowIdx % gridSize;
      const int j = rowIdx / gridSize;
      if( ( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 ) && localIdx == 0 )
      {
         columnIdx = rowIdx;
         value = 1.0;
      }
      else
      {
         switch( localIdx )
         {
            case 0:
               columnIdx = rowIdx - gridSize;
               value = 1.0;
               break;
            case 1:
               columnIdx = rowIdx - 1;
               value = 1.0;
               break;
            case 2:
               columnIdx = rowIdx;
               value = -4.0;
               break;
            case 3:
               columnIdx = rowIdx + 1;
               value = 1.0;
               break;
            case 4:
               columnIdx = rowIdx + gridSize;
               value = 1.0;
               break;
         }
      }
   };
   matrix.forElements( 0, matrixSize, f );
}

template< typename Device >
void laplaceOperatorMultidiagonalMatrix()
{
   std::cout << " Sparse matrix test:" << std::endl;
   for( int gridSize = 16; gridSize <= 8192; gridSize *= 2 )
   {
      std::cout << "  Grid size = " << gridSize << std::endl;
      TNL::Timer timer;

      std::cout << "   setElement on host: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::MultidiagonalMatrix< float, Device, int > matrix;
         setElement_on_host( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      if( std::is_same< Device, TNL::Devices::Cuda >::value )
      {
         std::cout << "   setElement on host and transfer on GPU: ";
         timer.reset();
         timer.start();
         for( int i = 0; i < testsCount; i++ )
         {
            TNL::Matrices::MultidiagonalMatrix< float, Device, int > matrix;
            setElement_on_host_and_transfer( gridSize, matrix );
         }
         timer.stop();
         std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;
      }

      std::cout << "   setElement on device: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::MultidiagonalMatrix< float, Device, int > matrix;
         setElement_on_device( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   getRow: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::MultidiagonalMatrix< float, Device, int > matrix;
         getRow( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   forElements: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::MultidiagonalMatrix< float, Device, int > matrix;
         forElements( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

   }
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ... " << std::endl;
   laplaceOperatorMultidiagonalMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating Laplace operator matrix on CUDA GPU ... " << std::endl;
   laplaceOperatorMultidiagonalMatrix< TNL::Devices::Cuda >();
#endif
}
