// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber,
//                 Jakub Klinkovsky,
//                 Vit Hanousek

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/TNL/Cuda/StreamPool.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser.h>
#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

namespace noaTNL {
namespace Meshes {

/****
 * 1D traverser, host
 */
template< typename Real,
          typename Index >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
GridTraverser< Meshes::Grid< 1, Real, Devices::Host, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   UserData& userData,
   GridTraverserMode mode,
   const int& stream )
{
   GridEntity entity( *gridPointer );
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer );

      entity.getCoordinates() = begin;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
      entity.getCoordinates() = end - 1;
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
   else
   {
#ifdef HAVE_OPENMP
      if( Devices::Host::isOMPEnabled() && end.x() - begin.x() > 512 )
      {
#pragma omp parallel firstprivate( begin, end )
         {
            GridEntity entity( *gridPointer );
#pragma omp for
            // TODO: g++ 5.5 crashes when coding this loop without auxiliary x as bellow
            for( IndexType x = begin.x(); x < end.x(); x++ )
            {
               entity.getCoordinates().x() = x;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }
         }
      }
      else
      {
         GridEntity entity( *gridPointer );
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() < end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
      }
#else
      GridEntity entity( *gridPointer );
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
 * 1D traverser, CUDA
 */
#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void
GridTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( coordinates < end )
   {
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
}

template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void
GridBoundaryTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   if( threadIdx.x == 0 )
   {
      coordinates.x() = begin.x();
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
   if( threadIdx.x == 1 )
   {
      coordinates.x() = end.x() - 1;
      GridEntity entity( *grid, coordinates );
      entity.refresh();
      EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
}

#endif

template< typename Real,
          typename Index >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities >
void
GridTraverser< Meshes::Grid< 1, Real, Devices::Cuda, Index > >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   UserData& userData,
   GridTraverserMode mode,
   const int& stream )
{
#ifdef HAVE_CUDA
   auto& pool = Cuda::StreamPool::getInstance();
   const cudaStream_t& s = pool.getStream( stream );

   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
   if( processOnlyBoundaryEntities )
   {
      dim3 cudaBlockSize( 2 );
      dim3 cudaBlocks( 1 );
      GridBoundaryTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              userData,
              begin,
              end );
   }
   else
   {
      dim3 blockSize( 256 ), blocksCount, gridsCount;
      Cuda::setupThreads(
         blockSize,
         blocksCount,
         gridsCount,
         end.x() - begin.x() );
      dim3 gridIdx;
      for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
      {
         dim3 gridSize;
         Cuda::setupGrid(
            blocksCount,
            gridsCount,
            gridIdx,
            gridSize );
         GridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< blocksCount, blockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              userData,
              begin,
              end,
              gridIdx.x );
      }

      /*dim3 cudaBlockSize( 256 );
      dim3 cudaBlocks;
      cudaBlocks.x = Cuda::getNumberOfBlocks( end.x() - begin.x(), cudaBlockSize.x );
      const IndexType cudaXGrids = Cuda::getNumberOfGrids( cudaBlocks.x );

      for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
         GridTraverser1D< Real, Index, GridEntity, UserData, EntitiesProcessor >
            <<< cudaBlocks, cudaBlockSize, 0, s >>>
            ( &gridPointer.template getData< Devices::Cuda >(),
              userData,
              begin,
              end,
              gridXIdx );*/
   }

#ifdef NDEBUG
   if( mode == synchronousMode )
   {
      cudaStreamSynchronize( s );
      TNL_CHECK_CUDA_DEVICE;
   }
#else
   cudaStreamSynchronize( s );
   TNL_CHECK_CUDA_DEVICE;
#endif

#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace Meshes
} // namespace noaTNL
