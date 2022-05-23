#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Cuda/StreamPool.h>

namespace TNL {

/****
 * This is only a helper class for Traverser specializations for Grid.
 */
template< typename Grid, typename Cell >
class GridTraverser
{
};

/****
 * 2D grid, Devices::Host
 */
template< typename Real,
          typename Index,
          typename Cell >
class GridTraverser< Meshes::Grid< 2, Real, Devices::Host, Index >, Cell >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Devices::Host, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using RealType = Real;
      using DeviceType = Devices::Host;
      using IndexType = Index;
      using CoordinatesType = typename GridType::CoordinatesType;

      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         UserData& userData,
         // FIXME: hack around nvcc bug (error: default argument not at end of parameter list)
//         const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
};

/****
 * 2D grid, Devices::Cuda
 */
template< typename Real,
          typename Index,
          typename Cell >
class GridTraverser< Meshes::Grid< 2, Real, Devices::Cuda, Index >, Cell >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Devices::Cuda, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using RealType = Real;
      using DeviceType = Devices::Cuda;
      using IndexType = Index;
      using CoordinatesType = typename GridType::CoordinatesType;

      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         UserData& userData,
         // FIXME: hack around nvcc bug (error: default argument not at end of parameter list)
//         const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
};

} // namespace TNL

#include "GridTraverser_impl.h"

