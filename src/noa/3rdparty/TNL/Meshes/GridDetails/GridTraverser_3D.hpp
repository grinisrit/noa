// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/TNL/Cuda/StreamPool.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser.h>
#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Meshes {


/****
 * 3D traverser, host
 */
template< typename Real,
          typename Index >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary,
      int ZOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 3, Real, Devices::Host, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   UserData& userData,
   GridTraverserMode mode,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer, begin, gridEntityParameters... );

      if( ZOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() < end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() < end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().z() = begin.z();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().z() = end.z() - 1;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
                 entity.getCoordinates().z() < end.z();
                 entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() < end.x();
                 entity.getCoordinates().x() ++ )
            {
               entity.getCoordinates().y() = begin.y();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().y() = end.y() - 1;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().z() = begin.z();
              entity.getCoordinates().z() < end.z();
              entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().y() = begin.y();
                 entity.getCoordinates().y() < end.y();
                 entity.getCoordinates().y() ++ )
            {
               entity.getCoordinates().x() = begin.x();
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               entity.getCoordinates().x() = end.x() - 1;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
   }
   else
   {
#ifdef HAVE_OPENMP
      if( Devices::Host::isOMPEnabled() )
      {
#pragma omp parallel firstprivate( begin, end )
         {
            GridEntity entity( *gridPointer, begin, gridEntityParameters... );
#pragma omp for
            // TODO: g++ 5.5 crashes when coding this loop without auxiliary x and y as bellow
            for( IndexType z = begin.z(); z < end.z(); z ++ )
               for( IndexType y = begin.y(); y < end.y(); y ++ )
                  for( IndexType x = begin.x(); x < end.x(); x ++ )
                  {
                     entity.getCoordinates().x() = x;
                     entity.getCoordinates().y() = y;
                     entity.getCoordinates().z() = z;
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
                  }
         }
      }
      else
      {
         GridEntity entity( *gridPointer, begin, gridEntityParameters... );
         for( entity.getCoordinates().z() = begin.z();
              entity.getCoordinates().z() < end.z();
              entity.getCoordinates().z() ++ )
            for( entity.getCoordinates().y() = begin.y();
                 entity.getCoordinates().y() < end.y();
                 entity.getCoordinates().y() ++ )
               for( entity.getCoordinates().x() = begin.x();
                    entity.getCoordinates().x() < end.x();
                    entity.getCoordinates().x() ++ )
                  {
                     entity.refresh();
                     EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
                  }
      }
#else
      GridEntity entity( *gridPointer, begin, gridEntityParameters... );
      for( entity.getCoordinates().z() = begin.z();
           entity.getCoordinates().z() < end.z();
           entity.getCoordinates().z() ++ )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() < end.y();
              entity.getCoordinates().y() ++ )
            for( entity.getCoordinates().x() = begin.x();
                 entity.getCoordinates().x() < end.x();
                 entity.getCoordinates().x() ++ )
               {
                  entity.refresh();
                  EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
               }
#endif
   }
}

/****
 * 3D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void
GridTraverser3D(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = begin.y() + Cuda::getGlobalThreadIdx_y( gridIdx );
   coordinates.z() = begin.z() + Cuda::getGlobalThreadIdx_z( gridIdx );

   if( coordinates < end )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      if( ! processOnlyBoundaryEntities || entity.isBoundaryEntity() )
      {
         EntitiesProcessor::processEntity
         ( *grid,
           userData,
           entity );
      }
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void
GridTraverser3DBoundaryAlongXY(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginX,
   const Index endX,
   const Index beginY,
   const Index endY,
   const Index fixedZ,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = beginX + Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = beginY + Cuda::getGlobalThreadIdx_y( gridIdx );
   coordinates.z() = fixedZ;

   if( coordinates.x() < endX && coordinates.y() < endY )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      EntitiesProcessor::processEntity
      ( *grid,
        userData,
        entity );
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void
GridTraverser3DBoundaryAlongXZ(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginX,
   const Index endX,
   const Index beginZ,
   const Index endZ,
   const Index fixedY,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = beginX + Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = fixedY;
   coordinates.z() = beginZ + Cuda::getGlobalThreadIdx_y( gridIdx );

   if( coordinates.x() < endX && coordinates.z() < endZ )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      EntitiesProcessor::processEntity
      ( *grid,
        userData,
        entity );
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor,
          bool processOnlyBoundaryEntities,
          typename... GridEntityParameters >
__global__ void
GridTraverser3DBoundaryAlongYZ(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginY,
   const Index endY,
   const Index beginZ,
   const Index endZ,
   const Index fixedX,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = fixedX;
   coordinates.y() = beginY + Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.z() = beginZ + Cuda::getGlobalThreadIdx_y( gridIdx );

   if( coordinates.y() < endY && coordinates.z() < endZ )
   {
      GridEntity entity( *grid, coordinates, gridEntityParameters... );
      entity.refresh();
      EntitiesProcessor::processEntity
      ( *grid,
        userData,
        entity );
   }
}
#endif

template< typename Real,
          typename Index >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary,
         int YOrthogonalBoundary,
         int ZOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 3, Real, Devices::Cuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   UserData& userData,
   GridTraverserMode mode,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
#ifdef HAVE_CUDA
   if( processOnlyBoundaryEntities &&
       ( GridEntity::getEntityDimension() == 3 || GridEntity::getEntityDimension() == 0 ) )
   {
      dim3 cudaBlockSize( 16, 16 );
      const IndexType entitiesAlongX = end.x() - begin.x();
      const IndexType entitiesAlongY = end.y() - begin.y();
      const IndexType entitiesAlongZ = end.z() - begin.z();

      dim3 cudaBlocksCountAlongXY, cudaBlocksCountAlongXZ, cudaBlocksCountAlongYZ,
           cudaGridsCountAlongXY, cudaGridsCountAlongXZ, cudaGridsCountAlongYZ;

      Cuda::setupThreads( cudaBlockSize, cudaBlocksCountAlongXY, cudaGridsCountAlongXY, entitiesAlongX, entitiesAlongY );
      Cuda::setupThreads( cudaBlockSize, cudaBlocksCountAlongXZ, cudaGridsCountAlongXZ, entitiesAlongX, entitiesAlongZ - 2 );
      Cuda::setupThreads( cudaBlockSize, cudaBlocksCountAlongYZ, cudaGridsCountAlongYZ, entitiesAlongY - 2, entitiesAlongZ - 2 );

      auto& pool = Cuda::StreamPool::getInstance();
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();

      const cudaStream_t& s1 = pool.getStream( stream );
      const cudaStream_t& s2 = pool.getStream( stream + 1 );
      const cudaStream_t& s3 = pool.getStream( stream + 2 );
      const cudaStream_t& s4 = pool.getStream( stream + 3 );
      const cudaStream_t& s5 = pool.getStream( stream + 4 );
      const cudaStream_t& s6 = pool.getStream( stream + 5 );

      dim3 gridIdx, gridSize;
      for( gridIdx.y = 0; gridIdx.y < cudaGridsCountAlongXY.y; gridIdx.y++ )
         for( gridIdx.x = 0; gridIdx.x < cudaGridsCountAlongXY.x; gridIdx.x++ )
         {
            Cuda::setupGrid( cudaBlocksCountAlongXY, cudaGridsCountAlongXY, gridIdx, gridSize );
            GridTraverser3DBoundaryAlongXY< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongXY, cudaBlockSize, 0 , s1 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.x(),
                    end.x(),
                    begin.y(),
                    end.y(),
                    begin.z(),
                    gridIdx,
                    gridEntityParameters... );
            GridTraverser3DBoundaryAlongXY< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongXY, cudaBlockSize, 0, s2 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.x(),
                    end.x(),
                    begin.y(),
                    end.y(),
                    end.z(),
                    gridIdx,
                    gridEntityParameters... );
         }
      for( gridIdx.y = 0; gridIdx.y < cudaGridsCountAlongXZ.y; gridIdx.y++ )
         for( gridIdx.x = 0; gridIdx.x < cudaGridsCountAlongXZ.x; gridIdx.x++ )
         {
            Cuda::setupGrid( cudaBlocksCountAlongXZ, cudaGridsCountAlongXZ, gridIdx, gridSize );
            GridTraverser3DBoundaryAlongXZ< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongXZ, cudaBlockSize, 0, s3 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.x(),
                    end.x(),
                    begin.z() + 1,
                    end.z() - 1,
                    begin.y(),
                    gridIdx,
                    gridEntityParameters... );
            GridTraverser3DBoundaryAlongXZ< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongXZ, cudaBlockSize, 0, s4 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.x(),
                    end.x(),
                    begin.z() + 1,
                    end.z() - 1,
                    end.y(),
                    gridIdx,
                    gridEntityParameters... );
         }
      for( gridIdx.y = 0; gridIdx.y < cudaGridsCountAlongYZ.y; gridIdx.y++ )
         for( gridIdx.x = 0; gridIdx.x < cudaGridsCountAlongYZ.x; gridIdx.x++ )
         {
            Cuda::setupGrid( cudaBlocksCountAlongYZ, cudaGridsCountAlongYZ, gridIdx, gridSize );
            GridTraverser3DBoundaryAlongYZ< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongYZ, cudaBlockSize, 0, s5 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.y() + 1,
                    end.y() - 1,
                    begin.z() + 1,
                    end.z() - 1,
                    begin.x(),
                    gridIdx,
                    gridEntityParameters... );
            GridTraverser3DBoundaryAlongYZ< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< cudaBlocksCountAlongYZ, cudaBlockSize, 0, s6 >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin.y() + 1,
                    end.y() - 1,
                    begin.z() + 1,
                    end.z() - 1,
                    end.x(),
                    gridIdx,
                    gridEntityParameters... );
         }
      cudaStreamSynchronize( s1 );
      cudaStreamSynchronize( s2 );
      cudaStreamSynchronize( s3 );
      cudaStreamSynchronize( s4 );
      cudaStreamSynchronize( s5 );
      cudaStreamSynchronize( s6 );
      TNL_CHECK_CUDA_DEVICE;
   }
   else
   {
      dim3 cudaBlockSize( 8, 8, 8 );
      dim3 cudaBlocksCount, cudaGridsCount;

      Cuda::setupThreads( cudaBlockSize, cudaBlocksCount, cudaGridsCount,
                          end.x() - begin.x(),
                          end.y() - begin.y(),
                          end.z() - begin.z() );

      auto& pool = Cuda::StreamPool::getInstance();
      const cudaStream_t& s = pool.getStream( stream );

      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      dim3 gridIdx, gridSize;
      for( gridIdx.z = 0; gridIdx.z < cudaGridsCount.z; gridIdx.z ++ )
         for( gridIdx.y = 0; gridIdx.y < cudaGridsCount.y; gridIdx.y ++ )
            for( gridIdx.x = 0; gridIdx.x < cudaGridsCount.x; gridIdx.x ++ )
            {
               Cuda::setupGrid( cudaBlocksCount, cudaGridsCount, gridIdx, gridSize );
               GridTraverser3D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
                  <<< gridSize, cudaBlockSize, 0, s >>>
                  ( &gridPointer.template getData< Devices::Cuda >(),
                    userData,
                    begin,
                    end,
                    gridIdx,
                    gridEntityParameters... );
            }

      // only launches into the stream 0 are synchronized
      if( stream == 0 )
      {
         cudaStreamSynchronize( s );
         TNL_CHECK_CUDA_DEVICE;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Meshes
} // namespace noa::TNL
