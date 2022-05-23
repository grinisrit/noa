// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NeighborGridEntitiesStorage.h>

namespace noa::TNL {
namespace Meshes {

template< typename GridEntity, int NeighborEntityDimension, typename StencilStorage >
class NeighborGridEntityGetter;

template< typename GridEntityType >
class BoundaryGridEntityChecker;

template< typename GridEntityType >
class GridEntityCenterGetter;

template< typename Grid, int EntityDimension, typename Config >
class GridEntity
{};

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >
{
public:
   using GridType = Meshes::Grid< Dimension, Real, Device, Index >;
   using MeshType = GridType;
   using RealType = typename GridType::RealType;
   using IndexType = typename GridType::IndexType;
   using CoordinatesType = typename GridType::CoordinatesType;
   using ConfigType = Config;

   constexpr static int
   getMeshDimension()
   {
      return GridType::getMeshDimension();
   };

   constexpr static int
   getEntityDimension()
   {
      return EntityDimension;
   };

   using EntityOrientationType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using EntityBasisType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using PointType = typename GridType::PointType;

   using NeighborGridEntitiesStorageType = NeighborGridEntitiesStorage< GridEntity, Config >;

   template< int NeighborEntityDimension = getEntityDimension() >
   using NeighborEntities =
      NeighborGridEntityGetter< GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >,
                                NeighborEntityDimension >;

   __cuda_callable__
   inline GridEntity( const GridType& grid );

   __cuda_callable__
   inline GridEntity( const GridType& grid,
                      const CoordinatesType& coordinates,
                      const EntityOrientationType& orientation,
                      const EntityBasisType& basis );

   __cuda_callable__
   inline const CoordinatesType&
   getCoordinates() const;

   __cuda_callable__
   inline CoordinatesType&
   getCoordinates();

   __cuda_callable__
   inline void
   setCoordinates( const CoordinatesType& coordinates );

   /***
    * Call this method every time the coordinates are changed
    * to recompute the mesh entity index. The reason for this strange
    * mechanism is a performance.
    */
   __cuda_callable__
   inline
      // void setIndex( IndexType entityIndex );
      void
      refresh();

   __cuda_callable__
   inline Index
   getIndex() const;

   __cuda_callable__
   inline const EntityOrientationType&
   getOrientation() const;

   __cuda_callable__
   inline void
   setOrientation( const EntityOrientationType& orientation );

   __cuda_callable__
   inline const EntityBasisType&
   getBasis() const;

   __cuda_callable__
   inline void
   setBasis( const EntityBasisType& basis );

   template< int NeighborEntityDimension = getEntityDimension() >
   __cuda_callable__
   inline const NeighborEntities< NeighborEntityDimension >&
   getNeighborEntities() const;

   __cuda_callable__
   inline bool
   isBoundaryEntity() const;

   __cuda_callable__
   inline PointType
   getCenter() const;

   __cuda_callable__
   inline const RealType&
   getMeasure() const;

   __cuda_callable__
   inline const GridType&
   getMesh() const;

protected:
   const GridType& grid;

   IndexType entityIndex;

   CoordinatesType coordinates;

   EntityOrientationType orientation;

   EntityBasisType basis;

   NeighborGridEntitiesStorageType neighborEntitiesStorage;

   //__cuda_callable__ inline
   // GridEntity();

   friend class BoundaryGridEntityChecker< GridEntity >;

   friend class GridEntityCenterGetter< GridEntity >;
};

/****
 * Specializations for cells
 */
template< int Dimension, typename Real, typename Device, typename Index, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >
{
public:
   using GridType = Meshes::Grid< Dimension, Real, Device, Index >;
   using MeshType = GridType;
   using RealType = typename GridType::RealType;
   using IndexType = typename GridType::IndexType;
   using CoordinatesType = typename GridType::CoordinatesType;
   using PointType = typename GridType::PointType;
   using ConfigType = Config;

   constexpr static int
   getMeshDimension()
   {
      return GridType::getMeshDimension();
   };

   constexpr static int
   getEntityDimension()
   {
      return getMeshDimension();
   };

   using EntityOrientationType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using EntityBasisType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using NeighborGridEntitiesStorageType = NeighborGridEntitiesStorage< GridEntity, Config >;

   template< int NeighborEntityDimension = getEntityDimension() >
   using NeighborEntities =
      NeighborGridEntityGetter< GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >,
                                NeighborEntityDimension >;

   __cuda_callable__
   inline GridEntity( const GridType& grid );

   __cuda_callable__
   inline GridEntity( const GridType& grid,
                      const CoordinatesType& coordinates,
                      const EntityOrientationType& orientation = EntityOrientationType( (Index) 0 ),
                      const EntityBasisType& basis = EntityBasisType( (Index) 1 ) );

   __cuda_callable__
   inline const CoordinatesType&
   getCoordinates() const;

   __cuda_callable__
   inline CoordinatesType&
   getCoordinates();

   __cuda_callable__
   inline void
   setCoordinates( const CoordinatesType& coordinates );

   /***
    * Call this method every time the coordinates are changed
    * to recompute the mesh entity index. The reason for this strange
    * mechanism is a performance.
    */
   __cuda_callable__
   inline
      // void setIndex( IndexType entityIndex );
      void
      refresh();

   __cuda_callable__
   inline Index
   getIndex() const;

   __cuda_callable__
   inline const EntityOrientationType
   getOrientation() const;

   __cuda_callable__
   inline void
   setOrientation( const EntityOrientationType& orientation ){};

   __cuda_callable__
   inline const EntityBasisType
   getBasis() const;

   __cuda_callable__
   inline void
   setBasis( const EntityBasisType& basis ){};

   template< int NeighborEntityDimension = Dimension >
   __cuda_callable__
   inline const NeighborEntities< NeighborEntityDimension >&
   getNeighborEntities() const;

   __cuda_callable__
   inline bool
   isBoundaryEntity() const;

   __cuda_callable__
   inline PointType
   getCenter() const;

   __cuda_callable__
   inline const RealType&
   getMeasure() const;

   __cuda_callable__
   inline const PointType&
   getEntityProportions() const;

   __cuda_callable__
   inline const GridType&
   getMesh() const;

protected:
   const GridType& grid;

   IndexType entityIndex;

   CoordinatesType coordinates;

   NeighborGridEntitiesStorageType neighborEntitiesStorage;

   //__cuda_callable__ inline
   // GridEntity();

   friend class BoundaryGridEntityChecker< GridEntity >;

   friend class GridEntityCenterGetter< GridEntity >;
};

/****
 * Specialization for vertices
 */
template< int Dimension, typename Real, typename Device, typename Index, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >
{
public:
   using GridType = Meshes::Grid< Dimension, Real, Device, Index >;
   using MeshType = GridType;
   using RealType = typename GridType::RealType;
   using IndexType = typename GridType::IndexType;
   using CoordinatesType = typename GridType::CoordinatesType;
   using PointType = typename GridType::PointType;
   using ConfigType = Config;

   constexpr static int
   getMeshDimension()
   {
      return GridType::getMeshDimension();
   };

   constexpr static int
   getEntityDimension()
   {
      return 0;
   };

   using EntityOrientationType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using EntityBasisType = Containers::StaticVector< getMeshDimension(), IndexType >;
   using NeighborGridEntitiesStorageType = NeighborGridEntitiesStorage< GridEntity, Config >;

   template< int NeighborEntityDimension = getEntityDimension() >
   using NeighborEntities = NeighborGridEntityGetter< GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >,
                                                      NeighborEntityDimension >;

   __cuda_callable__
   inline GridEntity( const GridType& grid );

   __cuda_callable__
   inline GridEntity( const GridType& grid,
                      const CoordinatesType& coordinates,
                      const EntityOrientationType& orientation = EntityOrientationType( (Index) 0 ),
                      const EntityBasisType& basis = EntityOrientationType( (Index) 0 ) );

   __cuda_callable__
   inline const CoordinatesType&
   getCoordinates() const;

   __cuda_callable__
   inline CoordinatesType&
   getCoordinates();

   __cuda_callable__
   inline void
   setCoordinates( const CoordinatesType& coordinates );

   /***
    * Call this method every time the coordinates are changed
    * to recompute the mesh entity index. The reason for this strange
    * mechanism is a performance.
    */
   __cuda_callable__
   inline
      // void setIndex( IndexType entityIndex );
      void
      refresh();

   __cuda_callable__
   inline Index
   getIndex() const;

   __cuda_callable__
   inline const EntityOrientationType
   getOrientation() const;

   __cuda_callable__
   inline void
   setOrientation( const EntityOrientationType& orientation ){};

   __cuda_callable__
   inline const EntityBasisType
   getBasis() const;

   __cuda_callable__
   inline void
   setBasis( const EntityBasisType& basis ){};

   template< int NeighborEntityDimension = getEntityDimension() >
   __cuda_callable__
   inline const NeighborEntities< NeighborEntityDimension >&
   getNeighborEntities() const;

   __cuda_callable__
   inline bool
   isBoundaryEntity() const;

   __cuda_callable__
   inline PointType
   getCenter() const;

   // compatibility with meshes, equivalent to getCenter
   __cuda_callable__
   inline PointType
   getPoint() const;

   __cuda_callable__
   inline const RealType
   getMeasure() const;

   __cuda_callable__
   inline PointType
   getEntityProportions() const;

   __cuda_callable__
   inline const GridType&
   getMesh() const;

protected:
   const GridType& grid;

   IndexType entityIndex;

   CoordinatesType coordinates;

   NeighborGridEntitiesStorageType neighborEntitiesStorage;

   friend class BoundaryGridEntityChecker< GridEntity >;

   friend class GridEntityCenterGetter< GridEntity >;
};

}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/GridEntity_impl.h>
