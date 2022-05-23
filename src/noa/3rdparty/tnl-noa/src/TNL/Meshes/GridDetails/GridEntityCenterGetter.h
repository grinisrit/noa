// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Meshes {

template< typename GridEntity >
class GridEntityCenterGetter
{};

/***
 * 1D grids
 */
template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config > >
{
public:
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 1, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x() );
   }
};

template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config > >
{
public:
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 0, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x() );
   }
};

/****
 * 2D grids
 */
template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config > >
{
public:
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 2, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
                        grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y() );
   }
};

template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1, Config > >
{
public:
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 1, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType(
         grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceSteps().x(),
         grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceSteps().y() );
   }
};

template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0, Config > >
{
public:
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 0, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + entity.getCoordinates().x() * grid.getSpaceSteps().x(),
                        grid.getOrigin().y() + entity.getCoordinates().y() * grid.getSpaceSteps().y() );
   }
};

/***
 * 3D grid
 */
template< typename Real, typename Device, typename Index, int EntityDimension, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, EntityDimension, Config > >
{
public:
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, EntityDimension, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType(
         grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * grid.getSpaceSteps().x(),
         grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * grid.getSpaceSteps().y(),
         grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 * entity.getBasis().z() ) * grid.getSpaceSteps().z() );
   }
};

template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config > >
{
public:
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 3, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + ( entity.getCoordinates().x() + 0.5 ) * grid.getSpaceSteps().x(),
                        grid.getOrigin().y() + ( entity.getCoordinates().y() + 0.5 ) * grid.getSpaceSteps().y(),
                        grid.getOrigin().z() + ( entity.getCoordinates().z() + 0.5 ) * grid.getSpaceSteps().z() );
   }
};

template< typename Real, typename Device, typename Index, typename Config >
class GridEntityCenterGetter< GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0, Config > >
{
public:
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   using GridEntityType = GridEntity< GridType, 0, Config >;
   using PointType = typename GridType::PointType;

   __cuda_callable__
   inline static PointType
   getEntityCenter( const GridEntityType& entity )
   {
      const GridType& grid = entity.getMesh();
      return PointType( grid.getOrigin().x() + ( entity.getCoordinates().x() ) * grid.getSpaceSteps().x(),
                        grid.getOrigin().y() + ( entity.getCoordinates().y() ) * grid.getSpaceSteps().y(),
                        grid.getOrigin().z() + ( entity.getCoordinates().z() ) * grid.getSpaceSteps().z() );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
