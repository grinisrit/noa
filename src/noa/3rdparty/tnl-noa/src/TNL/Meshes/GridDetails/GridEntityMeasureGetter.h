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

template< typename Grid, int EntityDimension >
class GridEntityMeasureGetter;

/***
 * Common implementation for vertices
 */
template< int Dimension, typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< Dimension, Real, Device, Index >, 0 >
{
public:
   using GridType = Grid< Dimension, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      return 0.0;
   }
};

/****
 * 1D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 1, Real, Device, Index >, 1 >
{
public:
   using GridType = Grid< 1, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      return grid.template getSpaceStepsProducts< 1 >();
   }
};

/****
 * 2D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 2, Real, Device, Index >, 2 >
{
public:
   using GridType = Grid< 2, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      return grid.template getSpaceStepsProducts< 1, 1 >();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 2, Real, Device, Index >, 1 >
{
public:
   using GridType = Grid< 2, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      if( entity.getOrientation() == 0 )
         return grid.template getSpaceStepsProducts< 1, 0 >();

      return grid.template getSpaceStepsProducts< 0, 1 >();
   }
};

/****
 * 3D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 3, Real, Device, Index >, 3 >
{
public:
   using GridType = Grid< 3, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      return grid.template getSpaceStepsProducts< 1, 1, 1 >();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 3, Real, Device, Index >, 2 >
{
public:
   using GridType = Grid< 3, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      if( entity.getOrientation() == 0 )
         return grid.template getSpaceStepsProducts< 1, 1, 0 >();

      if( entity.getOrientation() == 1 )
         return grid.template getSpaceStepsProducts< 1, 0, 1 >();

      return grid.template getSpaceStepsProducts< 0, 1, 1 >();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityMeasureGetter< Meshes::Grid< 3, Real, Device, Index >, 1 >
{
public:
   using GridType = Grid< 3, Real, Device, Index >;

   template< typename EntityType >
   __cuda_callable__
   inline static Real
   getMeasure( const GridType& grid, const EntityType& entity )
   {
      if( entity.getOrientation() == 0 )
         return grid.template getSpaceStepsProducts< 1, 0, 0 >();

      if( entity.getOrientation() == 1 )
         return grid.template getSpaceStepsProducts< 0, 1, 0 >();

      return grid.template getSpaceStepsProducts< 0, 0, 1 >();
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
