// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Meshes/Traverser.h>

#include "GridTraverserBenchmarkHelper.h"
#include "AddOneEntitiesProcessor.h"
#include "BenchmarkTraverserUserData.h"
#include "SimpleCell.h"

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename GridEntity,
          typename UserData,
          typename EntitiesProcessor >
__global__ void
_GridTraverser1D(
   const Meshes::Grid< 1, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const Index gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
   //typename GridType::CoordinatesType coordinates;

   GridEntity entity( *grid );
   entity.getCoordinates().x() = begin.x() + ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   //coordinates.x() = begin.x() + ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( entity.getCoordinates() <= end )
   {
      entity.refresh();
      //( userData.u->getData() )[ entity.getIndex( coordinates ) ] += ( RealType ) 1.0;
      //( userData.u->getData() )[ coordinates.x() ] += ( RealType ) 1.0;
      //userData.data[ entity.getIndex() ] += ( RealType ) 1.0;
      //userData.u->getData()[ entity.getIndex() ] += ( RealType ) 1.0;
      ( *userData.u )( entity ) += ( RealType ) 1.0;
      //EntitiesProcessor::processEntity( entity.getMesh(), userData, entity );
   }
}
#endif

template< typename Real,
          typename Index >
class GridTraverserBenchmarkHelper< Meshes::Grid< 1, Real, Devices::Host, Index > >
{
   public:

      constexpr static int Dimension = 1;
      using GridType = Meshes::Grid< Dimension, Real, Devices::Host, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using RealType = typename GridType::RealType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;
      using MeshFunction = Functions::MeshFunctionView< GridType >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using CellType = typename GridType::template EntityType< Dimension, Meshes::GridEntityNoStencilStorage >;
      using SimpleCellType = SimpleCell< GridType >;
      using Traverser = Meshes::Traverser< GridType, CellType >;
      using UserDataType = BenchmarkTraverserUserData< MeshFunction >;
      using AddOneEntitiesProcessorType = AddOneEntitiesProcessor< UserDataType >;

      static void simpleCellTest( const GridPointer& grid,
                                     UserDataType& userData,
                                     std::size_t size )
      {
         const CoordinatesType begin( 0 );
         const CoordinatesType end = CoordinatesType( size ) - CoordinatesType( 1 );
         SimpleCellType entity( *grid );
         for( entity.getCoordinates().x() = begin.x();
              entity.getCoordinates().x() <= end.x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            //userData.u->getData()[ entity.getIndex() ] += ( RealType ) 1.0;
            ( *userData.u )( entity ) += ( RealType ) 1.0;
         }

      }
};

template< typename Real,
          typename Index >
class GridTraverserBenchmarkHelper< Meshes::Grid< 1, Real, Devices::Cuda, Index > >
{
   public:

      constexpr static int Dimension = 1;
      using GridType = Meshes::Grid< Dimension, Real, Devices::Cuda, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using RealType = typename GridType::RealType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;
      using MeshFunction = Functions::MeshFunctionView< GridType >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using CellType = typename GridType::template EntityType< Dimension, Meshes::GridEntityNoStencilStorage >;
      using SimpleCellType = SimpleCell< GridType >;
      using Traverser = Meshes::Traverser< GridType, CellType >;
      using UserDataType = BenchmarkTraverserUserData< MeshFunction >;
      using AddOneEntitiesProcessorType = AddOneEntitiesProcessor< UserDataType >;

      static void simpleCellTest( const GridPointer& grid,
                                  UserDataType& userData,
                                  std::size_t size )
      {
#ifdef HAVE_CUDA
            dim3 blockSize( 256 ), blocksCount, gridsCount;
            Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size );
            dim3 gridIdx;
            for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
            {
               dim3 gridSize;
               Cuda::setupGrid(
                  blocksCount,
                  gridsCount,
                  gridIdx,
                  gridSize );
               _GridTraverser1D< RealType, IndexType, SimpleCellType, UserDataType, AddOneEntitiesProcessorType >
               <<< blocksCount, blockSize >>>
               ( &grid.template getData< Devices::Cuda >(),
                 userData,
                 CoordinatesType( 0 ),
                 CoordinatesType( size ) - CoordinatesType( 1 ),
                 gridIdx.x );

            }
#endif
      }
};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
