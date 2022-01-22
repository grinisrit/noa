// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid1D.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid2D.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid3D.h>

namespace noaTNL {
namespace Meshes {

/****
 * 1D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity,
          int EntityDimension >
class GridEntityGetter<
   Meshes::Grid< 1, Real, Device, Index >,
   GridEntity,
   EntityDimension >
{
   public:
 
      static constexpr int entityDimension = EntityDimension;
 
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );
         return GridEntity
            ( grid,
              CoordinatesType( index ),
              typename GridEntity::EntityOrientationType( 0 ),
              typename GridEntity::EntityBasisType( EntityDimension ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + CoordinatesType( 1 - entityDimension ), "wrong coordinates" );
         return entity.getCoordinates().x();
      }
};

/****
 * 2D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
 
      static constexpr int entityDimension = 2;
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );

         const CoordinatesType dimensions = grid.getDimensions();

         return GridEntity
            ( grid,
              CoordinatesType( index % dimensions.x(),
                               index / dimensions.x() ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 1, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions(), "wrong coordinates" );

         //const CoordinatesType coordinates = entity.getCoordinates();
         //const CoordinatesType dimensions = grid.getDimensions();
 
         return entity.getCoordinates().y() * grid.getDimensions().x() + entity.getCoordinates().x();
      }
 
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
 
      static constexpr int entityDimension = 1;
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension, EntityConfig > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );
 
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % aux, index / aux ),
                 typename GridEntity::EntityOrientationType( 1, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxFaces;
         const IndexType& aux = dimensions.x();
         return GridEntity
            ( grid,
              CoordinatesType( i % aux, i / aux ),
              typename GridEntity::EntityOrientationType( 0, 1 ),
              typename GridEntity::EntityBasisType( 1, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + abs( entity.getOrientation() ), "wrong coordinates" );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         if( entity.getOrientation().x() )
            return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
         return grid.numberOfNxFaces + coordinates.y() * dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
 
      static constexpr int entityDimension = 0;
 
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );

         const CoordinatesType dimensions = grid.getDimensions();

         const IndexType aux = dimensions.x() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( index % aux,
                               index / aux ),
              typename GridEntity::EntityOrientationType( 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), grid.getDimensions(), "wrong coordinates" );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return coordinates.y() * ( dimensions.x() + 1 ) + coordinates.x();
      }
};

/****
 * 3D grid
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 3 >
{
   public:
 
      static constexpr int entityDimension = 3;
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );

         const CoordinatesType dimensions = grid.getDimensions();

         return GridEntity
            ( grid,
              CoordinatesType( index % dimensions.x(),
                               ( index / dimensions.x() ) % dimensions.y(),
                               index / ( dimensions.x() * dimensions.y() ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 1, 1, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions(), "wrong coordinates" );

         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return ( coordinates.z() * dimensions.y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 2 >
{
   public:
 
      static constexpr int entityDimension = 2;
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );

         const CoordinatesType dimensions = grid.getDimensions();
 
         if( index < grid.numberOfNxFaces )
         {
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % aux,
                                  ( index / aux ) % dimensions.y(),
                                  index / ( aux * dimensions.y() ) ),
                 typename GridEntity::EntityOrientationType( 1, 0, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1, 1 ) );
         }
         if( index < grid.numberOfNxAndNyFaces )
         {
            const IndexType i = index - grid.numberOfNxFaces;
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( i % dimensions.x(),
                                  ( i / dimensions.x() ) % aux,
                                  i / ( aux * dimensions.x() ) ),
                 typename GridEntity::EntityOrientationType( 0, 1, 0 ),
                 typename GridEntity::EntityBasisType( 1, 0, 1 ) );
         }
         const IndexType i = index - grid.numberOfNxAndNyFaces;
         return GridEntity
            ( grid,
              CoordinatesType( i % dimensions.x(),
                               ( i / dimensions.x() ) % dimensions.y(),
                               i / ( dimensions.x() * dimensions.y() ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 1 ),
              typename GridEntity::EntityBasisType( 1, 1, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), grid.getDimensions() + abs( entity.getOrientation() ), "wrong coordinates" );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();

 
         if( entity.getOrientation().x() )
         {
            return ( coordinates.z() * dimensions.y() + coordinates.y() ) *
               ( dimensions.x() + 1 ) + coordinates.x();
         }
         if( entity.getOrientation().y() )
         {
            return grid.numberOfNxFaces +
               ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) *
               dimensions.x() + coordinates.x();
         }
         return grid.numberOfNxAndNyFaces +
            ( coordinates.z() * dimensions.y() + coordinates.y() ) *
            dimensions.x() + coordinates.x();
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 1 >
{
   public:
 
      static constexpr int entityDimension = 1;
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );
 
         const CoordinatesType dimensions = grid.getDimensions();

         if( index < grid.numberOfDxEdges )
         {
            const IndexType aux = dimensions.y() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( index % dimensions.x(),
                                  ( index / dimensions.x() ) % aux,
                                  index / ( dimensions.x() * aux ) ),
                 typename GridEntity::EntityOrientationType( 0, 0, 0 ),
                 typename GridEntity::EntityBasisType( 1, 0, 0 ) );

         }
         if( index < grid.numberOfDxAndDyEdges )
         {
            const IndexType i = index - grid.numberOfDxEdges;
            const IndexType aux = dimensions.x() + 1;
            return GridEntity
               ( grid,
                 CoordinatesType( i % aux,
                                  ( i / aux ) % dimensions.y(),
                                  i / ( aux * dimensions.y() ) ),
                 typename GridEntity::EntityOrientationType( 0, 0, 0 ),
                 typename GridEntity::EntityBasisType( 0, 1, 0 ) );
         }
         const IndexType i = index - grid.numberOfDxAndDyEdges;
         const IndexType aux1 = dimensions.x() + 1;
         const IndexType aux2 = dimensions.y() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( i % aux1,
                               ( i / aux1 ) % aux2,
                               i / ( aux1 * aux2 ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0, 1 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(),
                        grid.getDimensions() + CoordinatesType( 1, 1, 1 ) - entity.getBasis(),
                        "wrong coordinates" );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         if( entity.getBasis().x() )
            return ( coordinates.z() * ( dimensions.y() + 1 ) +
                     coordinates.y() ) * dimensions.x() + coordinates.x();
         if( entity.getBasis().y() )
            return grid.numberOfDxEdges +
               ( coordinates.z() * dimensions.y() + coordinates.y() ) * ( dimensions.x() + 1 ) +
               coordinates.x();
         return grid.numberOfDxAndDyEdges +
            ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) * ( dimensions.x() + 1 ) +
            coordinates.x();

      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class GridEntityGetter< Meshes::Grid< 3, Real, Device, Index >, GridEntity, 0 >
{
   public:
 
      static constexpr int entityDimension = 0;
 
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      //typedef typename GridType::template GridEntity< entityDimension > GridEntity;
 
      __cuda_callable__ inline
      static GridEntity getEntity( const GridType& grid,
                                   const IndexType& index )
      {
         TNL_ASSERT_GE( index, 0, "Index must be non-negative." );
         TNL_ASSERT_LT( index, grid.template getEntitiesCount< GridEntity >(), "Index is out of bounds." );

         const CoordinatesType dimensions = grid.getDimensions();
 
         const IndexType auxX = dimensions.x() + 1;
         const IndexType auxY = dimensions.y() + 1;
         return GridEntity
            ( grid,
              CoordinatesType( index % auxX,
                               ( index / auxX ) % auxY,
                               index / ( auxX * auxY ) ),
              typename GridEntity::EntityOrientationType( 0, 0, 0 ),
              typename GridEntity::EntityBasisType( 0, 0, 0 ) );
      }
 
      __cuda_callable__ inline
      static IndexType getEntityIndex( const GridType& grid,
                                       const GridEntity& entity )
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), grid.getDimensions(), "wrong coordinates" );
 
         const CoordinatesType coordinates = entity.getCoordinates();
         const CoordinatesType dimensions = grid.getDimensions();
 
         return ( coordinates.z() * ( dimensions.y() + 1 ) + coordinates.y() ) *
                ( dimensions.x() + 1 ) +
                coordinates.x();
      }
};

} // namespace Meshes
} // namespace noaTNL
