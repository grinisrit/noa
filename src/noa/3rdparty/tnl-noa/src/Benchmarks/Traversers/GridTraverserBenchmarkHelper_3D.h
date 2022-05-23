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
_GridTraverser3D(
   const Meshes::Grid< 3, Real, Devices::Cuda, Index >* grid,
   UserData userData,
   const typename GridEntity::CoordinatesType begin,
   const typename GridEntity::CoordinatesType end,
   const dim3 gridIdx )
{
   typedef Real RealType;
   typedef Index IndexType;
   typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;

   GridEntity entity( *grid );
   entity.getCoordinates().x() = begin.x() + ( gridIdx.x * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   entity.getCoordinates().y() = begin.y() + ( gridIdx.y * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
   entity.getCoordinates().z() = begin.z() + ( gridIdx.z * Cuda::getMaxGridSize() + blockIdx.z ) * blockDim.z + threadIdx.z;

   if( entity.getCoordinates() <= end )
   {
      entity.refresh();
      ( *userData.u )( entity ) += ( RealType ) 1.0;
   }
}
#endif

template< typename Real,
          typename Index >
class GridTraverserBenchmarkHelper< Meshes::Grid< 3, Real, Devices::Host, Index > >
{
   public:

      constexpr static int Dimension = 3;
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
         for( entity.getCoordinates().z() = begin.z();
              entity.getCoordinates().z() <= end.z();
              entity.getCoordinates().z()++ )
            for( entity.getCoordinates().y() = begin.y();
                 entity.getCoordinates().y() <= end.y();
                 entity.getCoordinates().y()++ )
               for( entity.getCoordinates().x() = begin.x();
                    entity.getCoordinates().x() <= end.x();
                    entity.getCoordinates().x() ++ )
                  {
                     entity.refresh();
                     ( *userData.u )( entity ) += ( RealType ) 1.0;
                  }
      }
};

template< typename Real,
          typename Index >
class GridTraverserBenchmarkHelper< Meshes::Grid< 3, Real, Devices::Cuda, Index > >
{
   public:

      constexpr static int Dimension = 3;
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
            dim3 blockSize( 32, 4, 2 ), blocksCount, gridsCount;
            Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.z = 0; gridIdx.z < gridsCount.z; gridIdx.z++ )
               for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
                  for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
                  {
                     dim3 gridSize;
                     Cuda::setupGrid(
                        blocksCount,
                        gridsCount,
                        gridIdx,
                        gridSize );
                     _GridTraverser3D< RealType, IndexType, SimpleCellType, UserDataType, AddOneEntitiesProcessorType >
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
