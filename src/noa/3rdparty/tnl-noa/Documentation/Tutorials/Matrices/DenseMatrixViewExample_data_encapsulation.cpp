#include <iostream>
#ifdef HAVE_CUDA
#include <cuda.h>
#endif
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void encapsulation()
{
   const int size = 5;

   /***
    * Allocate the dense matrix with no use of TNL
    */
   double* host_data = new double[ size * size ];
   for( int row = 0; row < size; row++ )
      for( int column = 0; column < size; column++ )
         host_data[ row * size + column ] = row * size + column + 1;
   double* data = nullptr;
   if( std::is_same< Device, TNL::Devices::Host >::value )
   {
      data = new double[ size * size ];
      memcpy( data, host_data, sizeof( double ) * size * size );
   }
#ifdef HAVE_CUDA
   else if( std::is_same< Device, TNL::Devices::Cuda >::value )
   {
      cudaMalloc( ( void**) &data, sizeof( double ) * size * size );
      cudaMemcpy( data, host_data, sizeof( double ) * size * size,  cudaMemcpyHostToDevice );
   }
#endif

   /***
    * Encapsulate the matrix into DenseMatrixView.
    */
   TNL::Containers::VectorView< double, Device > dataView( data, size * size );
   TNL::Matrices::DenseMatrixView< double, Device, int, TNL::Algorithms::Segments::RowMajorOrder > matrix( 5, 5, dataView );

   std::cout << "Dense matrix view reads as:" << std::endl;
   std::cout << matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i ) mutable {
      matrix.setElement( i, i, -i );
   };
   TNL::Algorithms::ParallelFor< Device >::exec( 0, 5, f );

   std::cout << "Dense matrix view after elements manipulation:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Do not forget to free allocated memory :)
    */
   delete[] host_data;
   if( std::is_same< Device, TNL::Devices::Host >::value )
      delete[] data;
#ifdef HAVE_CUDA
   else if( std::is_same< Device, TNL::Devices::Cuda >::value )
      cudaFree( data );
#endif
}

int main( int argc, char* argv[] )
{
   std::cout << "Dense matrix encapsulation on host:" << std::endl;
   encapsulation< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Dense matrix encapsulation on CUDA device:" << std::endl;
   encapsulation< TNL::Devices::Cuda >();
#endif
}

