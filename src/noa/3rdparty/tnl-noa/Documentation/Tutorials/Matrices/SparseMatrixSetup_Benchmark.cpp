#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Timer.h>

const int testsCount = 5;

template< typename Matrix >
void STL_Map( const int gridSize, Matrix& matrix )
{
   /***
    * Set  matrix representing approximation of the Laplace operator on regular
    * grid using the finite difference method by means of STL map.
    */
   const int matrixSize = gridSize * gridSize;
   matrix.setDimensions( matrixSize, matrixSize );
   std::map< std::pair< int, int >, double > map;
   for( int j = 0; j < gridSize; j++ )
      for( int i = 0; i < gridSize; i++ )
      {
         const int rowIdx = j * gridSize + i;
         if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx ),  1.0 ) );
         else
         {
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx - gridSize ),  1.0 ) );
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx - 1 ),  1.0 ) );
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx ),  -4.0 ) );
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx + 1 ),  1.0 ) );
            map.insert( std::make_pair( std::make_pair( rowIdx, rowIdx + gridSize ),  1.0 ) );
         }
      }
   matrix.setElements( map );
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
   TNL::Containers::Vector< int, typename Matrix::DeviceType, int > rowCapacities( matrixSize, 5 );
   matrix.setDimensions( matrixSize, matrixSize );
   matrix.setRowCapacities( rowCapacities );

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
   using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;

   const int matrixSize = gridSize * gridSize;
   TNL::Containers::Vector< int, typename HostMatrix::DeviceType, int > rowCapacities( matrixSize, 5 );
   HostMatrix hostMatrix( matrixSize, matrixSize );
   hostMatrix.setRowCapacities( rowCapacities );

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
   TNL::Containers::Vector< int, typename Matrix::DeviceType, int > rowCapacities( matrixSize, 5 );
   matrix.setDimensions( matrixSize, matrixSize );
   matrix.setRowCapacities( rowCapacities );

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
   TNL::Containers::Vector< int, typename Matrix::DeviceType, int > rowCapacities( matrixSize, 5 );
   matrix.setDimensions( matrixSize, matrixSize );
   matrix.setRowCapacities( rowCapacities );

   auto matrixView = matrix.getView();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      const int i = rowIdx % gridSize;
      const int j = rowIdx / gridSize;
      auto row = matrixView.getRow( rowIdx );
      if( i == 0 || j == 0 || i == gridSize - 1 || j == gridSize - 1 )
         row.setElement( 2, rowIdx,  1.0 );
      else
      {
         row.setElement( 0, rowIdx - gridSize, 1.0 );
         row.setElement( 1, rowIdx - 1, 1.0 );
         row.setElement( 2, rowIdx, -4.0 );
         row.setElement( 3, rowIdx + 1, 1.0 );
         row.setElement( 4, rowIdx + gridSize, 1.0 );
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
   TNL::Containers::Vector< int, typename Matrix::DeviceType, int > rowCapacities( matrixSize, 5 );
   matrix.setDimensions( matrixSize, matrixSize );
   matrix.setRowCapacities( rowCapacities );

   auto f = [=] __cuda_callable__ ( int rowIdx, int localIdx, int& columnIdx, float& value ) mutable {
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
void laplaceOperatorSparseMatrix()
{
   std::cout << " Sparse matrix test:" << std::endl;
   for( int gridSize = 16; gridSize <= 8192; gridSize *= 2 )
   {
      std::cout << "  Grid size = " << gridSize << std::endl;
      TNL::Timer timer;

      std::cout << "   STL map: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::SparseMatrix< float, Device, int > matrix;
         STL_Map( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   setElement on host: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::SparseMatrix< float, Device, int > matrix;
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
            TNL::Matrices::SparseMatrix< float, Device, int > matrix;
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
         TNL::Matrices::SparseMatrix< float, Device, int > matrix;
         setElement_on_device( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   getRow: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::SparseMatrix< float, Device, int > matrix;
         getRow( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

      std::cout << "   forElements: ";
      timer.reset();
      timer.start();
      for( int i = 0; i < testsCount; i++ )
      {
         TNL::Matrices::SparseMatrix< float, Device, int > matrix;
         forElements( gridSize, matrix );
      }
      timer.stop();
      std::cout << timer.getRealTime() / ( double ) testsCount << " sec." << std::endl;

   }
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ... " << std::endl;
   laplaceOperatorSparseMatrix< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Creating Laplace operator matrix on CUDA GPU ... " << std::endl;
   laplaceOperatorSparseMatrix< TNL::Devices::Cuda >();
#endif
}
