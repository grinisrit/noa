// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntityConfig.h>
#include <TNL/Meshes/Traverser.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Pointers/SharedPointer.h>
#include "cuda-kernels.h"
#include "AddOneEntitiesProcessor.h"
#include "AddTwoEntitiesProcessor.h"
#include "BenchmarkTraverserUserData.h"
#include "GridTraversersBenchmark.h"
#include "GridTraverserBenchmarkHelper.h"
#include "SimpleCell.h"

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark< 2, Device, Real, Index >
{
   public:

      using Vector = Containers::Vector< Real, Device, Index >;
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using Coordinates = typename GridType::CoordinatesType;
      using MeshFunction = Functions::MeshFunctionView< GridType >;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;
      using CellType = typename GridType::template EntityType< 2, Meshes::GridEntityNoStencilStorage >;
      using SimpleCellType = SimpleCell< GridType >;
      using Traverser = Meshes::Traverser< GridType, CellType >;
      using UserDataType = BenchmarkTraverserUserData< MeshFunction >;
      using AddOneEntitiesProcessorType = AddOneEntitiesProcessor< UserDataType >;
      using AddTwoEntitiesProcessorType = AddTwoEntitiesProcessor< UserDataType >;

      GridTraversersBenchmark( Index size )
      :size( size ),
       v( size * size ),
       grid( size, size ),
       userData( u )
      {
         v_data = v.getData();
         u->bind( grid, v );
      }

      void reset()
      {
         v.setValue( 0.0 );
      };

      void addOneUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
               for( int j = 0; j < size; j++ )
                  v_data[ i * size + j ] += (Real) 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 16, 16 ), blocksCount, gridsCount;
            Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  fullGridTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
#endif
         }
      }

      void addOneUsingParallelFor()
      {
         Index _size = this->size;
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            data[ j * _size + i ] += (Real) 1.0;
         };

         Algorithms::ParallelFor2D< Device, Algorithms::AsynchronousMode >::exec(
            ( Index ) 0,
            ( Index ) 0,
            this->size,
            this->size,
            f, v.getData() );
      }

      void addOneUsingSimpleCell()
      {
         /*const GridType* currentGrid = &grid.template getData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            SimpleCellType entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.getCoordinates().y() = j;
            entity.refresh();
            data[ entity.getIndex() ] += (Real) 1.0;
         };

         Algorithms::ParallelFor2D< Device, Algorithms::AsynchronousMode >::exec(
            ( Index ) 0,
            ( Index ) 0,
            this->size,
            this->size,
            f, v.getData() );*/
         GridTraverserBenchmarkHelper< GridType >::simpleCellTest(
            grid,
            userData,
            size );

      }

      void addOneUsingParallelForAndMeshFunction()
      {
         const GridType* currentGrid = &grid.template getData< Device >();
         MeshFunction* _u = &u.template modifyData< Device >();
         auto f = [=] __cuda_callable__ ( Index i, Index j,  Real* data )
         {
            SimpleCellType entity( *currentGrid );
            entity.getCoordinates().x() = i;
            entity.getCoordinates().y() = j;
            entity.refresh();
            //( *_u )( entity ) += (Real) 1.0;
            _u->getData().getData()[ entity.getIndex() ] += (Real) 1.0;
         };

         Algorithms::ParallelFor2D< Device, Algorithms::AsynchronousMode >::exec(
            ( Index ) 0,
            ( Index ) 0,
            this->size,
            this->size,
            f, v.getData() );
      }


      void addOneUsingTraverser()
      {
         using CoordinatesType = typename GridType::CoordinatesType;
         traverser.template processAllEntities< AddOneEntitiesProcessorType >
            ( grid, userData );

         /*Meshes::GridTraverser< Grid >::template processEntities< Cell, WriteOneEntitiesProcessorType, WriteOneTraverserUserDataType, false >(
           grid,
           CoordinatesType( 0 ),
           grid->getDimensions() - CoordinatesType( 1 ),
           userData );*/
         /*const CoordinatesType begin( 0 );
         const CoordinatesType end = CoordinatesType( size ) - CoordinatesType( 1 );
         MeshFunction* _u = &u.template modifyData< Device >();
         Cell entity( *grid );
         for( Index y = begin.y(); y <= end.y(); y ++ )
            for( Index x = begin.x(); x <= end.x(); x ++ )
            {
               entity.getCoordinates().x() = x;
               entity.getCoordinates().y() = y;
               entity.refresh();
               WriteOneEntitiesProcessorType::processEntity( entity.getMesh(), userData, entity );
            }*/
      }

      bool checkAddOne( int loops, bool reseting )
      {
         if( reseting )
            return Algorithms::containsOnlyValue( v, 1.0 );
         return Algorithms::containsOnlyValue( v, ( Real ) loops );
      }

      void traverseUsingPureC()
      {
         if( std::is_same< Device, Devices::Host >::value )
         {
            for( int i = 0; i < size; i++ )
            {
               v_data[ i * size ] += (Real) 2.0;
               v_data[ i * size + size - 1 ] += (Real) 2.0;
            }
            for( int j = 1; j < size - 1; j++ )
            {
               v_data[ j ] += (Real) 2.0;
               v_data[ ( size - 1 ) * size + j ] += (Real) 2.0;
            }

            for( int i = 1; i < size - 1; i++ )
               for( int j = 1; j < size - 1; j++ )
                  v_data[ i * size + j ] += (Real) 1.0;
         }
         else // Device == Devices::Cuda
         {
#ifdef HAVE_CUDA
            dim3 blockSize( 32, 8 ), blocksCount, gridsCount;
            Cuda::setupThreads(
               blockSize,
               blocksCount,
               gridsCount,
               size,
               size );
            dim3 gridIdx;
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  boundariesTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
            for( gridIdx.y = 0; gridIdx.y < gridsCount.y; gridIdx.y++ )
               for( gridIdx.x = 0; gridIdx.x < gridsCount.x; gridIdx.x++ )
               {
                  dim3 gridSize;
                  Cuda::setupGrid(
                     blocksCount,
                     gridsCount,
                     gridIdx,
                     gridSize );
                  interiorTraverseKernel2D<<< gridSize, blockSize >>>( size, gridIdx, v_data );
               }
#endif
         }
      }

      void traverseUsingTraverser()
      {
         //traverser.template processAllEntities< AddOneEntitiesProcessorType >
         traverser.template processBoundaryEntities< AddTwoEntitiesProcessorType >
            ( grid, userData );
         traverser.template processInteriorEntities< AddOneEntitiesProcessorType >
            ( grid, userData );
      }

   protected:

      Index size;
      Vector v;
      Real* v_data;
      GridPointer grid;
      MeshFunctionPointer u;
      Traverser traverser;
      UserDataType userData;
};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
