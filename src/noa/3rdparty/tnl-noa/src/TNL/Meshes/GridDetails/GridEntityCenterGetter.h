// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>

namespace noa::TNL {
namespace Meshes {

template< int Dimension, typename Real, typename Device, typename Index >
class Grid;

template< class, int >
class GridEntity;

template< class >
class GridEntityCenterGetter;

/***
 * 1D grids
 */
template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1 > >
{
public:
   using Grid = Meshes::Grid< 1, Real, Device, Index >;
   using Entity = GridEntity< Grid, 1 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x() );
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0 > >
{
public:
   using Grid = Meshes::Grid< 1, Real, Device, Index >;
   using Entity = GridEntity< Grid, 0 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x() );
   }
};

/****
 * 2D grids
 */
template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2 > >
{
public:
   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, 2 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
                    grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y() );
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1 > >
{
public:
   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, 1 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point(
         grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 * ! entity.getNormals().x() ) * grid.getSpaceSteps().x(),
         grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 * ! entity.getNormals().y() ) * grid.getSpaceSteps().y() );
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0 > >
{
public:
   using Grid = Meshes::Grid< 2, Real, Device, Index >;
   using Entity = GridEntity< Grid, 0 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + entity.getCoordinates().x() * grid.getSpaceSteps().x(),
                    grid.getOrigin().y() + entity.getCoordinates().y() * grid.getSpaceSteps().y() );
   }
};

/***
 * 3D grid
 */
template< typename Real, typename Device, typename Index, int EntityDimension >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, EntityDimension > >
{
public:
   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, EntityDimension >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point(
         grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 * ! entity.getNormals().x() ) * grid.getSpaceSteps().x(),
         grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 * ! entity.getNormals().y() ) * grid.getSpaceSteps().y(),
         grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 * ! entity.getNormals().z() ) * grid.getSpaceSteps().z() );
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3 > >
{
public:
   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, 3 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
                    grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y(),
                    grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 ) * grid.getSpaceSteps().z() );
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0 > >
{
public:
   using Grid = Meshes::Grid< 3, Real, Device, Index >;
   using Entity = GridEntity< Grid, 0 >;
   using Point = typename Grid::PointType;

   __cuda_callable__
   inline static Point
   getEntityCenter( const Entity& entity )
   {
      const Grid& grid = entity.getMesh();
      return Point( grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x(),
                    grid.getOrigin().y() + ( entity.getCoordinates().y() ) * grid.getSpaceSteps().y(),
                    grid.getOrigin().z() + ( entity.getCoordinates().z() ) * grid.getSpaceSteps().z() );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
