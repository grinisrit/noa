// Implemented by: Tomas Oberhuber

#pragma once

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

#ifdef HAVE_CUDA

/****
 * Full grid traversing
 */
template< typename Real,
          typename Index >
__global__ void fullGridTraverseKernel1D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( threadIdx_x < size )
      v_data[ threadIdx_x ] += (Real) 1.0;
}

template< typename Real,
          typename Index >
__global__ void fullGridTraverseKernel2D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( threadIdx_x < size && threadIdx_y < size )
      v_data[ threadIdx_y * size + threadIdx_x ] += (Real) 1.0;
}

template< typename Real,
          typename Index >
__global__ void fullGridTraverseKernel3D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   const Index threadIdx_z = ( gridIdx.z * Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
   if( threadIdx_x < size && threadIdx_y < size && threadIdx_z < size )
      v_data[ ( threadIdx_z * size + threadIdx_y ) * size + threadIdx_x ] += (Real) 1.0;
}

/****
 * Traversing interior cells
 */
template< typename Real,
          typename Index >
__global__ void interiorTraverseKernel1D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( threadIdx_x > 0 && threadIdx_x < size - 1 )
      v_data[ threadIdx_x ] += (Real) 1.0;
}

template< typename Real,
          typename Index >
__global__ void interiorTraverseKernel2D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( threadIdx_x > 0 && threadIdx_y > 0 &&
       threadIdx_x < size - 1 && threadIdx_y < size - 1 )
         v_data[ threadIdx_y * size + threadIdx_x ] += (Real) 1.0;
}

template< typename Real,
          typename Index >
__global__ void interiorTraverseKernel3D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   const Index threadIdx_z = ( gridIdx.z * Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
   if( threadIdx_x > 0 && threadIdx_y > 0 && threadIdx_z > 0 &&
       threadIdx_x < size - 1 && threadIdx_y < size - 1 && threadIdx_z < size - 1 )
      v_data[ ( threadIdx_z * size + threadIdx_y ) * size + threadIdx_x ] += (Real) 1.0;
}

/****
 * Grid boundaries traversing
 */
template< typename Real,
          typename Index >
__global__ void boundariesTraverseKernel1D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( threadIdx_x == 0 || threadIdx_x == size - 1 )
      v_data[ threadIdx_x ] += (Real) 2.0;
}

template< typename Real,
          typename Index >
__global__ void boundariesTraverseKernel2D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   if( threadIdx_x > 0 && threadIdx_y > 0 &&
       threadIdx_x < size - 1 && threadIdx_y < size - 1 )
         v_data[ threadIdx_y * size + threadIdx_x ] += (Real) 2.0;
}

template< typename Real,
          typename Index >
__global__ void boundariesTraverseKernel3D( const Index size, const dim3 gridIdx, Real* v_data  )
{
   const Index threadIdx_x = ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index threadIdx_y = ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   const Index threadIdx_z = ( gridIdx.z * Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
   if( threadIdx_x == 0 || threadIdx_y == 0 || threadIdx_z == 0 ||
       threadIdx_x == size - 1 || threadIdx_y == size - 1 || threadIdx_z == size - 1 )
      v_data[ ( threadIdx_z * size + threadIdx_y ) * size + threadIdx_x ] += (Real) 2.0;
}

#endif
      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL

