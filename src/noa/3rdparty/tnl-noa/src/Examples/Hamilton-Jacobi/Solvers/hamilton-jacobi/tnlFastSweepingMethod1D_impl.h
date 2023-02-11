#pragma once

#include "tnlFastSweepingMethod.h"

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{

}

template< typename Real,
          typename Device,
          typename Index,
          typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >::
solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
       const MeshPointer& mesh,
       MeshFunctionPointer& Aux,
       const AnisotropyPointer& anisotropy,
       MeshFunctionPointer& u )
{
   MeshFunctionPointer auxPtr;
   InterfaceMapPointer interfaceMapPtr;
   auxPtr->setMesh( mesh );
   interfaceMapPtr->setMesh( mesh );
   std::cout << "Initiating the interface cells ..." << std::endl;
   BaseType::initInterface( u, auxPtr, interfaceMapPtr );

   auxPtr->write( "aux", "aux-ini.vti" );

   typename MeshType::Cell cell( *mesh );

   IndexType iteration( 0 );
   InterfaceMapType interfaceMap = *interfaceMapPtr;
   MeshFunctionType aux = *auxPtr;
   while( iteration < this->maxIterations )
   {
      if( std::is_same< DeviceType, Devices::Host >::value )
      {
          for( cell.getCoordinates().x() = 0;
               cell.getCoordinates().x() < mesh->getDimensions().x();
               cell.getCoordinates().x()++ )
             {
                cell.refresh();
                if( ! interfaceMap( cell ) )
                   this->updateCell( aux, cell );
             }


        for( cell.getCoordinates().x() = mesh->getDimensions().x() - 1;
             cell.getCoordinates().x() >= 0 ;
             cell.getCoordinates().x()-- )
           {
              cell.refresh();
              if( ! interfaceMap( cell ) )
                 this->updateCell( aux, cell );
           }
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
         // TODO: CUDA code
#ifdef __CUDACC__
          const int cudaBlockSize( 16 );
          int numBlocksX = Cuda::getNumberOfBlocks( mesh->getDimensions().x(), cudaBlockSize );
          dim3 blockSize( cudaBlockSize );
          dim3 gridSize( numBlocksX );

          BaseType ptr;



          bool* BlockIter = new bool[ numBlocksX ];

          bool *BlockIterDevice;
          cudaMalloc(&BlockIterDevice, ( numBlocksX ) * sizeof( bool ) );

          while( BlockIter[ 0 ] )
          {
           for( int i = 0; i < numBlocksX; i++ )
                BlockIter[ i ] = false;
           cudaMemcpy(BlockIterDevice, BlockIter, ( numBlocksX ) * sizeof( bool ), cudaMemcpyHostToDevice);

            CudaUpdateCellCaller<<< gridSize, blockSize >>>( ptr,
                                                             interfaceMapPtr.template getData< Device >(),
                                                             auxPtr.template modifyData< Device>(),
                                                             BlockIterDevice );
            cudaMemcpy(BlockIter, BlockIterDevice, ( numBlocksX ) * sizeof( bool ), cudaMemcpyDeviceToHost);

            for( int i = 1; i < numBlocksX; i++ )
                BlockIter[ 0 ] = BlockIter[ 0 ] || BlockIter[ i ];

          }
          delete[] BlockIter;
          cudaFree( BlockIterDevice );
          cudaDeviceSynchronize();

          TNL_CHECK_CUDA_DEVICE;

          aux = *auxPtr;
          interfaceMap = *interfaceMapPtr;
#endif
      }
      iteration++;
   }

   aux.write( "aux", "aux-final.vti" );
}

#ifdef __CUDACC__
template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > > ptr,
                                      const Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap,
                                      Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& aux,
                                      bool *BlockIterDevice )
{
    int thri = threadIdx.x;
    int blIdx = blockIdx.x;
    int i = thri + blockDim.x*blIdx;

    __shared__ volatile bool changed[16];
    changed[ thri ] = false;

    if( thri == 0 )
        changed[ 0 ] = true;

    const Meshes::Grid< 1, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
    __shared__ Real h;
    if( thri == 1 )
    {
        h = mesh.getSpaceSteps().x();
    }

    __shared__ volatile Real sArray[ 18 ];
    sArray[thri] = std::numeric_limits<  Real >::max();

    //filling sArray edges
    int dimX = mesh.getDimensions().x();
    __shared__ volatile int numOfBlockx;
    __shared__ int xkolik;
    if( thri == 0 )
    {
        xkolik = blockDim.x + 1;
        numOfBlockx = dimX/blockDim.x + ((dimX%blockDim.x != 0) ? 1:0);

        if( numOfBlockx - 1 == blIdx )
            xkolik = dimX - (blIdx)*blockDim.x+1;
    }
    __syncthreads();

    if( thri == 0 )
    {
        if( dimX > (blIdx+1) * blockDim.x )
            sArray[xkolik] = aux[ blIdx*blockDim.x - 1 + xkolik ];
        else
            sArray[xkolik] = std::numeric_limits< Real >::max();
    }

    if( thri == 1 )
    {
        if( blIdx != 0 )
            sArray[0] = aux[ blIdx*blockDim.x - 1 ];
        else
            sArray[0] = std::numeric_limits< Real >::max();
    }


    if( i < mesh.getDimensions().x() )
    {
        sArray[thri+1] = aux[ i ];
    }
    __syncthreads();

    while( changed[ 0 ] )
    {
        __syncthreads();

        changed[ thri ] = false;

    //calculation of update cell
        if( i < mesh.getDimensions().x() )
        {
            if( ! interfaceMap[ i ] )
            {
                changed[ thri ] = ptr.updateCell( sArray, thri+1, h );
            }
        }
        __syncthreads();



    //pyramid reduction
        if( thri < 8 ) changed[ thri ] = changed[ thri ] || changed[ thri + 8 ];
        if( thri < 4 ) changed[ thri ] = changed[ thri ] || changed[ thri + 4 ];
        if( thri < 2 ) changed[ thri ] = changed[ thri ] || changed[ thri + 2 ];
        if( thri < 1 ) changed[ thri ] = changed[ thri ] || changed[ thri + 1 ];
        __syncthreads();

        if( changed[ 0 ] && thri == 0 )
            BlockIterDevice[ blIdx ] = true;
        __syncthreads();
    }

    if( i < mesh.getDimensions().x()  && (!interfaceMap[ i ]) )
        aux[ i ] = sArray[ thri + 1 ];
}
#endif


