// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>

namespace noa::TNL {
namespace Meshes {

/****
 * This is only a helper class for Traverser specializations for Grid.
 */
template< typename Grid >
class GridTraverser
{
};

enum GridTraverserMode { synchronousMode, asynchronousMode };

/****
 * 1D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 1, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 1, Real, Devices::Host, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         UserData& userData,
         GridTraverserMode mode = synchronousMode, 
         const int& stream = 0 );
};

/****
 * 1D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 1, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities  >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         UserData& userData,
         GridTraverserMode mode = synchronousMode,
         const int& stream = 0 );
};


/****
 * 2D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 2, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 2, Real, Devices::Host, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
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
         //GridTraverserMode mode = synchronousMode,
         GridTraverserMode mode,
         // const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
};

/****
 * 2D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 2, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
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
         //GridTraverserMode mode = synchronousMode,
         GridTraverserMode mode,
         // const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
};


/****
 * 3D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 3, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 3, Real, Devices::Host, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         int ZOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         UserData& userData,
         // FIXME: hack around nvcc bug (error: default argument not at end of parameter list)
         //GridTraverserMode mode = synchronousMode,
         GridTraverserMode mode,
         // const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces and edges)
         const GridEntityParameters&... gridEntityParameters );
};

/****
 * 3D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 3, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
      typedef Pointers::SharedPointer<  GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         int ZOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         UserData& userData,
         // FIXME: hack around nvcc bug (error: default argument not at end of parameter list)
         //GridTraverserMode mode = synchronousMode,
         GridTraverserMode mode,
         // const int& stream = 0,
         const int& stream,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces and edges)
         const GridEntityParameters&... gridEntityParameters );
};

} // namespace Meshes
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser_1D.hpp>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser_2D.hpp>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser_3D.hpp>
