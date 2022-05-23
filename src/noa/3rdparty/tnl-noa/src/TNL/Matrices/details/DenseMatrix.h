// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Matrices {
namespace details {

template< typename Device >
class DenseDeviceDependentCode;
template<>
class DenseDeviceDependentCode< Devices::Host >
{
public:
   typedef Devices::Host Device;

   template< typename Real, typename Index, bool RowMajorOrder, typename RealAllocator, typename InVector, typename OutVector >
   static void
   vectorProduct( const DenseMatrixView< Real, Device, Index, RowMajorOrder >& matrix,
                  const InVector& inVector,
                  OutVector& outVector )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( Index row = 0; row < matrix.getRows(); row++ )
         outVector[ row ] = matrix.rowVectorProduct( row, inVector );
   }
};

template<>
class DenseDeviceDependentCode< Devices::Cuda >
{
public:
   typedef Devices::Cuda Device;

   template< typename Real, typename Index, bool RowMajorOrder, typename RealAllocator, typename InVector, typename OutVector >
   static void
   vectorProduct( const DenseMatrixView< Real, Device, Index, RowMajorOrder >& matrix,
                  const InVector& inVector,
                  OutVector& outVector )
   {
      MatrixVectorProductCuda( matrix, inVector, outVector );
   }
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator,
          typename Matrix1,
          typename Matrix2,
          int tileDim,
          int tileRowBlockSize >
__global__
void
DenseMatrixProductKernel( Dense< Real, Devices::Cuda, Index, RowMajorOrder >* resultMatrix,
                          const Matrix1* matrixA,
                          const Matrix2* matrixB,
                          const Real matrixAMultiplicator,
                          const Real matrixBMultiplicator,
                          const Index gridIdx_x,
                          const Index gridIdx_y )
{
   /****
    * Here we compute product C = A * B. To profit from the fast
    * shared memory we do it by tiles.
    */

   typedef Index IndexType;
   typedef Real RealType;
   __shared__ Real tileA[ tileDim * tileDim ];
   __shared__ Real tileB[ tileDim * tileDim ];
   __shared__ Real tileC[ tileDim * tileDim ];

   const IndexType& matrixARows = matrixA->getRows();
   const IndexType& matrixAColumns = matrixA->getColumns();
   const IndexType& matrixBRows = matrixB->getRows();
   const IndexType& matrixBColumns = matrixB->getColumns();

   /****
    * Reset the tile C
    */
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
      tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] = 0.0;

   /****
    * Compute the result tile coordinates
    */
   const IndexType resultTileRow = ( gridIdx_y * gridDim.y + blockIdx.y ) * tileDim;
   const IndexType resultTileColumn = ( gridIdx_x * gridDim.x + blockIdx.x ) * tileDim;

   /****
    * Sum over the matrix tiles
    */
   for( IndexType i = 0; i < matrixAColumns; i += tileDim ) {
      for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
         const IndexType matrixARow = resultTileRow + threadIdx.y + row;
         const IndexType matrixAColumn = i + threadIdx.x;
         if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
            tileA[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] =
               matrixAMultiplicator * matrixA->getElementFast( matrixARow, matrixAColumn );

         const IndexType matrixBRow = i + threadIdx.y + row;
         const IndexType matrixBColumn = resultTileColumn + threadIdx.x;
         if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
            tileB[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] =
               matrixBMultiplicator * matrixB->getElementFast( matrixBRow, matrixBColumn );
      }
      __syncthreads();

      const IndexType tileALastRow = tnlCudaMin( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = tnlCudaMin( tileDim, matrixAColumns - i );
      const IndexType tileBLastRow = tnlCudaMin( tileDim, matrixBRows - i );
      const IndexType tileBLastColumn = tnlCudaMin( tileDim, matrixBColumns - resultTileColumn );

      for( IndexType row = 0; row < tileALastRow; row += tileRowBlockSize ) {
         RealType sum( 0.0 );
         for( IndexType j = 0; j < tileALastColumn; j++ )
            sum += tileA[ ( threadIdx.y + row ) * tileDim + j ] * tileB[ j * tileDim + threadIdx.x ];
         tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] += sum;
      }
      __syncthreads();
   }

   /****
    * Write the result tile to the result matrix
    */
   const IndexType& matrixCRows = resultMatrix->getRows();
   const IndexType& matrixCColumns = resultMatrix->getColumns();
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
      const IndexType matrixCRow = resultTileRow + row + threadIdx.y;
      const IndexType matrixCColumn = resultTileColumn + threadIdx.x;
      if( matrixCRow < matrixCRows && matrixCColumn < matrixCColumns )
         resultMatrix->setElementFast( matrixCRow, matrixCColumn, tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] );
   }
}

template< typename Real,
          typename Index,
          typename Matrix,
          bool RowMajorOrder,
          typename RealAllocator,
          int tileDim,
          int tileRowBlockSize >
__global__
void
DenseTranspositionAlignedKernel( Dense< Real, Devices::Cuda, Index >* resultMatrix,
                                 const Matrix* inputMatrix,
                                 const Real matrixMultiplicator,
                                 const Index gridIdx_x,
                                 const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();

   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      tile[ Cuda::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
         inputMatrix->getElementFast( readColumnPosition, readRowPosition + rowBlock );
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      resultMatrix->setElementFast( writeColumnPosition,
                                    writeRowPosition + rowBlock,
                                    matrixMultiplicator
                                       * tile[ Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );
   }
}

template< typename Real,
          typename Index,
          bool RowMajorOrder,
          typename RealAllocator,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize >
__global__
void
DenseTranspositionNonAlignedKernel( Dense< Real, Devices::Cuda, Index >* resultMatrix,
                                    const Matrix* inputMatrix,
                                    const Real matrixMultiplicator,
                                    const Index gridIdx_x,
                                    const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();

   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   if( readColumnPosition < columns ) {
      const Index readOffset = readRowPosition * columns + readColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( readRowPosition + rowBlock < rows )
            tile[ Cuda::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix->getElementFast( readColumnPosition, readRowPosition + rowBlock );
      }
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   if( writeColumnPosition < rows ) {
      const Index writeOffset = writeRowPosition * rows + writeColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( writeRowPosition + rowBlock < columns )
            resultMatrix->setElementFast(
               writeColumnPosition,
               writeRowPosition + rowBlock,
               matrixMultiplicator * tile[ Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );
      }
   }
}

#endif

}  // namespace details
}  // namespace Matrices
}  // namespace noa::TNL