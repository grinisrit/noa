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

template< class, int >
class GridEntityGetter;

/****
 * 1D grid
 */
template< typename Real, typename Device, typename Index, int EntityDimension >
class GridEntityGetter< Grid< 1, Real, Device, Index >, EntityDimension >
{
public:
   static constexpr int entityDimension = EntityDimension;

   using GridType = Grid< 1, Real, Device, Index >;
   using EntityType = GridEntity< GridType, EntityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      return entity.getCoordinates().x();
   }
};

/****
 * 2D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 2, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using GridType = Grid< 2, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 2, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using GridType = Grid< 2, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType& coordinates = entity.getCoordinates();
      const CoordinatesType& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return coordinates.y() * ( dimensions.x() ) + coordinates.x();

      return grid.template getOrientedEntitiesCount< 1, 0 >() + coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 2, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using GridType = Grid< 2, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType& coordinates = entity.getCoordinates();
      const CoordinatesType& dimensions = grid.getDimensions();

      return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

/****
 * 3D grid
 */
template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 3, Real, Device, Index >, 3 >
{
public:
   static constexpr int entityDimension = 3;

   using GridType = Grid< 3, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType coordinates = entity.getCoordinates();
      const CoordinatesType dimensions = grid.getDimensions();

      return ( coordinates.z() * dimensions.y() + coordinates.y() ) * dimensions.x() + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 3, Real, Device, Index >, 2 >
{
public:
   static constexpr int entityDimension = 2;

   using GridType = Grid< 3, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType& coordinates = entity.getCoordinates();
      const CoordinatesType& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() ) + coordinates.x();

      if( entity.getOrientation() == 1 )
         return grid.template getOrientedEntitiesCount< 2, 0 >()
              + ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * dimensions.x() + coordinates.x();

      return grid.template getOrientedEntitiesCount< 2, 1 >() + grid.template getOrientedEntitiesCount< 2, 0 >()
           + ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 3, Real, Device, Index >, 1 >
{
public:
   static constexpr int entityDimension = 1;

   using GridType = Grid< 3, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType& coordinates = entity.getCoordinates();
      const CoordinatesType& dimensions = grid.getDimensions();

      if( entity.getOrientation() == 0 )
         return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * dimensions.x() + coordinates.x();

      if( entity.getOrientation() == 1 )
         return grid.template getOrientedEntitiesCount< 1, 0 >()
              + ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();

      return grid.template getOrientedEntitiesCount< 1, 1 >() + grid.template getOrientedEntitiesCount< 1, 0 >()
           + ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

template< typename Real, typename Device, typename Index >
class GridEntityGetter< Grid< 3, Real, Device, Index >, 0 >
{
public:
   static constexpr int entityDimension = 0;

   using GridType = Grid< 3, Real, Device, Index >;
   using EntityType = GridEntity< GridType, entityDimension >;
   using CoordinatesType = typename GridType::CoordinatesType;

   __cuda_callable__
   inline static Index
   getEntityIndex( const GridType& grid, const EntityType& entity )
   {
      TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
      TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + entity.getNormals(), "wrong coordinates" );

      const CoordinatesType coordinates = entity.getCoordinates();
      const CoordinatesType dimensions = grid.getDimensions();

      return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) + coordinates.x();
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
