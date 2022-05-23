#pragma once

#include "GridTraverser.h"

#include <TNL/Exceptions/CudaSupportMissing.h>

namespace TNL {

/****
 * 2D traverser, host
 */
template< typename Real,
          typename Index, 
          typename Cell >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
      int XOrthogonalBoundary,
      int YOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 2, Real, Devices::Host, Index >, Cell >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType begin,
   const CoordinatesType end,
   UserData& userData,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
   if( processOnlyBoundaryEntities )
   {
      GridEntity entity( *gridPointer, begin, gridEntityParameters... );
      
      if( YOrthogonalBoundary )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.getCoordinates().y() = begin.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().y() = end.y();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
      if( XOrthogonalBoundary )
         for( entity.getCoordinates().y() = begin.y();
              entity.getCoordinates().y() <= end.y();
              entity.getCoordinates().y() ++ )
         {
            entity.getCoordinates().x() = begin.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            entity.getCoordinates().x() = end.x();
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
         }
   }
   else
   {
      //TODO: This does not work with gcc-5.4 and older, should work at gcc 6.x
/*#pragma omp parallel for firstprivate( entity, begin, end ) if( Devices::Host::isOMPEnabled() )
      for( entity.getCoordinates().y() = begin.y();
           entity.getCoordinates().y() <= end.y();
           entity.getCoordinates().y() ++ )
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            EntitiesProcessor::processEntity( entity.getMesh(), *userDataPointer, entity );
         }*/
#ifdef HAVE_OPENMP
#pragma omp parallel firstprivate( begin, end ) if( Devices::Host::isOMPEnabled() )
#endif
      {
         GridEntity entity( *gridPointer, begin, gridEntityParameters... );
#ifdef HAVE_OPENMP
#pragma omp for 
#endif
         for( IndexType y = begin.y(); y <= end.y(); y ++ )
            for( IndexType x = begin.x(); x <= end.x(); x ++ )
            {
               entity.getCoordinates().x() = x;
               entity.getCoordinates().y() = y;
               entity.refresh();
               EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
            }      
      }
   }
}

/****
 * 2D traverser, CUDA
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
_GridTraverser2D(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData* userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + Cuda::getGlobalThreadIdx_x( gridIdx );
   coordinates.y() = begin.y() + Cuda::getGlobalThreadIdx_y( gridIdx );
   
   if( coordinates <= end )
   {
      //GridEntity entity( *grid, coordinates, gridEntityParameters... );
      //entity.refresh();
      /*if( ! processOnlyBoundaryEntities || 
         ( coordinates.x() == 0 || coordinates.y() == 0 ||
           coordinates.x() == grid->getDimensions().x() - 1 || coordinates.y() == grid->getDimensions().y() - 1 ) )*/
         //entity.isBoundaryEntity() )
      {
         EntitiesProcessor::processEntity
         ( *grid,
           *userData,
           coordinates.y() * grid->getDimensions().x() + coordinates.x(),
           coordinates
            );
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
_GridTraverser2DBoundary(
   const Meshes::Grid< 2, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const Index beginX,
   const Index endX,
   const Index beginY,
   const Index endY,
   const dim3 gridIdx,
   const GridEntityParameters... gridEntityParameters )
{
   using GridType = Meshes::Grid< 2, Real, Devices::Cuda, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   
   Index entitiesAlongX = endX - beginX + 1;
   Index entitiesAlongY = endY - beginY;
   
   Index threadId = Cuda::getGlobalThreadIdx_x( gridIdx );
   if( threadId < entitiesAlongX )
   {
      GridEntity entity( *grid, 
         CoordinatesType( beginX + threadId, beginY ),
         gridEntityParameters... );
      //printf( "X1: Thread %d -> %d %d x %d %d \n ", threadId, 
      //   entity.getCoordinates().x(), entity.getCoordinates().y(),
      //   grid->getDimensions().x(), grid->getDimensions().y() );
      entity.refresh();
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }
   else if( ( threadId -= entitiesAlongX ) < entitiesAlongX && threadId >= 0 )
   {
      GridEntity entity( *grid, 
         CoordinatesType( beginX + threadId, endY ),
         gridEntityParameters... );
      entity.refresh();
      //printf( "X2: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }
   else if( ( ( threadId -= entitiesAlongX ) < entitiesAlongY - 1 ) && threadId >= 0 )
   {
      GridEntity entity( *grid,
         CoordinatesType( beginX, beginY + threadId + 1 ),
      gridEntityParameters... );
      entity.refresh();
      //printf( "Y1: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );      
   }
   else if( ( ( threadId -= entitiesAlongY - 1 ) < entitiesAlongY - 1  ) && threadId >= 0 )
   {
      GridEntity entity( *grid,
         CoordinatesType( endX, beginY + threadId + 1 ),
      gridEntityParameters... );
      entity.refresh();
      //printf( "Y2: Thread %d -> %d %d \n ", threadId, entity.getCoordinates().x(), entity.getCoordinates().y() );
      EntitiesProcessor::processEntity( *grid, userData, entity );
   }
}

#endif

template< typename Real,
          typename Index,
          typename Cell >
   template<
      typename GridEntity,
      typename EntitiesProcessor,
      typename UserData,
      bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary,
         int YOrthogonalBoundary,
      typename... GridEntityParameters >
void
GridTraverser< Meshes::Grid< 2, Real, Devices::Cuda, Index >, Cell >::
processEntities(
   const GridPointer& gridPointer,
   const CoordinatesType& begin,
   const CoordinatesType& end,
   UserData& userData,
   const int& stream,
   const GridEntityParameters&... gridEntityParameters )
{
#ifdef HAVE_CUDA
   if( processOnlyBoundaryEntities && 
       ( GridEntity::getEntityDimension() == 2 || GridEntity::getEntityDimension() == 0 ) )
   {
      dim3 cudaBlockSize( 256 );      
      dim3 cudaBlocksCount, cudaGridsCount;
      IndexType cudaThreadsCount = 2 * ( end.x() - begin.x() + end.y() - begin.y() + 1 );
      Cuda::setupThreads( cudaBlockSize, cudaBlocksCount, cudaGridsCount, cudaThreadsCount );
      dim3 gridIdx, cudaGridSize;
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      for( gridIdx.x = 0; gridIdx.x < cudaGridsCount.x; gridIdx.x++ )
      {
         Cuda::setupGrid( cudaBlocksCount, cudaGridsCount, gridIdx, cudaGridSize );
         _GridTraverser2DBoundary< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 userData,
                 begin.x(),
                 end.x(),
                 begin.y(),
                 end.y(),
                 gridIdx,
                 gridEntityParameters... );
      }            
   }
   else
   {
      dim3 cudaBlockSize( 16, 16 );
      dim3 cudaBlocksCount, cudaGridsCount;
      Cuda::setupThreads( cudaBlockSize, cudaBlocksCount, cudaGridsCount,
                          end.x() - begin.x() + 1,
                          end.y() - begin.y() + 1 );
      
      auto& pool = Cuda::StreamPool::getInstance();
      const cudaStream_t& s = pool.getStream( stream );

      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      dim3 gridIdx, cudaGridSize;
      for( gridIdx.y = 0; gridIdx.y < cudaGridsCount.y; gridIdx.y ++ )
         for( gridIdx.x = 0; gridIdx.x < cudaGridsCount.x; gridIdx.x ++ )
         {
            Cuda::setupGrid( cudaBlocksCount, cudaGridsCount, gridIdx, cudaGridSize );
	    //Cuda::printThreadsSetup( cudaBlockSize, cudaBlocksCount, cudaGridSize, cudaGridsCount );
            TNL::_GridTraverser2D< Real, Index, GridEntity, UserData, EntitiesProcessor, processOnlyBoundaryEntities, GridEntityParameters... >
               <<< cudaGridSize, cudaBlockSize, 0, s >>>
               ( &gridPointer.template getData< Devices::Cuda >(),
                 &userData,
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

} // namespace TNL
