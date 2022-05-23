// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridEntityConfig.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>

namespace noa::TNL {
namespace Meshes {

template< typename GridEntity,
          int NeighborEntityDimension,
          typename GridEntityConfig,
          bool storage = GridEntityConfig::template neighborEntityStorage< GridEntity >( NeighborEntityDimension ) >
class NeighborGridEntityLayer : public NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig >
{
public:
   using BaseType = NeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1, GridEntityConfig >;
   using NeighborEntityGetterType = NeighborGridEntityGetter< GridEntity, NeighborEntityDimension >;

   using BaseType::getNeighborEntities;

   __cuda_callable__
   NeighborGridEntityLayer( const GridEntity& entity ) : BaseType( entity ), neighborEntities( entity ) {}

   __cuda_callable__
   const NeighborEntityGetterType&
   getNeighborEntities( const DimensionTag< NeighborEntityDimension >& tag ) const
   {
      return this->neighborEntities;
   }

   __cuda_callable__
   void
   refresh( const typename GridEntity::GridType& grid, const typename GridEntity::GridType::IndexType& entityIndex )
   {
      BaseType::refresh( grid, entityIndex );
      neighborEntities.refresh( grid, entityIndex );
   }

protected:
   NeighborEntityGetterType neighborEntities;
};

template< typename GridEntity, typename GridEntityConfig, bool storage >
class NeighborGridEntityLayer< GridEntity, 0, GridEntityConfig, storage >
{
public:
   using NeighborEntityGetterType = NeighborGridEntityGetter< GridEntity, 0 >;

   __cuda_callable__
   NeighborGridEntityLayer( const GridEntity& entity ) : neighborEntities( entity ) {}

   __cuda_callable__
   const NeighborEntityGetterType&
   getNeighborEntities( const DimensionTag< 0 >& tag ) const
   {
      return this->neighborEntities;
   }

   __cuda_callable__
   void
   refresh( const typename GridEntity::GridType& grid, const typename GridEntity::GridType::IndexType& entityIndex )
   {
      neighborEntities.refresh( grid, entityIndex );
   }

protected:
   NeighborEntityGetterType neighborEntities;
};

template< typename GridEntity, typename GridEntityConfig >
class NeighborGridEntitiesStorage
: public NeighborGridEntityLayer< GridEntity, GridEntity::getMeshDimension(), GridEntityConfig >
{
   using BaseType = NeighborGridEntityLayer< GridEntity, GridEntity::getMeshDimension(), GridEntityConfig >;

public:
   using BaseType::getNeighborEntities;
   using BaseType::refresh;

   __cuda_callable__
   NeighborGridEntitiesStorage( const GridEntity& entity ) : BaseType( entity ) {}

   template< int EntityDimension >
   __cuda_callable__
   const NeighborGridEntityGetter< GridEntity, EntityDimension >&
   getNeighborEntities() const
   {
      return BaseType::getNeighborEntities( DimensionTag< EntityDimension >() );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
