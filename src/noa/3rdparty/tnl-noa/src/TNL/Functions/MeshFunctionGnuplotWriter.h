// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {

namespace Meshes {
template< typename, typename, typename >
class MeshEntity;
}

namespace Functions {

class MeshFunctionGnuplotWriterBase
{
protected:
   template< typename Entity, int dim = Entity::getEntityDimension() >
   struct center
   {
      static auto
      get( const Entity& entity ) -> decltype( entity.getCenter() )
      {
         return entity.getCenter();
      }
   };

   template< typename Entity >
   struct center< Entity, 0 >
   {
      static auto
      get( const Entity& entity ) -> decltype( entity.getPoint() )
      {
         return entity.getPoint();
      }
   };

   template< typename MeshConfig, typename Device, typename Topology, int dim >
   struct center< TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >, dim >
   {
      static int
      get( const TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >& entity )
      {
         throw Exceptions::NotImplementedError();
      }
   };

   template< typename MeshConfig, typename Device, typename Topology >
   struct center< TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >, 0 >
   {
      static int
      get( const TNL::Meshes::MeshEntity< MeshConfig, Device, Topology >& entity )
      {
         throw Exceptions::NotImplementedError();
      }
   };
};

template< typename MeshFunction,
          typename Mesh = typename MeshFunction::MeshType,
          int EntitiesDimension = MeshFunction::getEntitiesDimension() >
class MeshFunctionGnuplotWriter : public MeshFunctionGnuplotWriterBase
{
public:
   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

   static bool
   write( const MeshFunction& function, std::ostream& str )
   {
      const MeshType& mesh = function.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         const EntityType& entity = mesh.template getEntity< EntityType >( i );
         typename MeshType::PointType v = center< EntityType >::get( entity );
         for( int j = 0; j < v.getSize(); j++ )
            str << v[ j ] << " ";
         str << function.getData().getElement( i ) << "\n";
      }
      return true;
   }
};

template< typename MeshFunction, typename Real, typename Device, typename Index, int EntityDimension >
class MeshFunctionGnuplotWriter< MeshFunction, Meshes::Grid< 2, Real, Device, Index >, EntityDimension >
: public MeshFunctionGnuplotWriterBase
{
public:
   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

   static bool
   write( const MeshFunction& function, std::ostream& str )
   {
      const MeshType& grid = function.getMesh();
      EntityType entity( grid );
      auto& c = entity.getCoordinates();
      for( c.y() = 0; c.y() < grid.getDimensions().y(); c.y()++ ) {
         for( c.x() = 0; c.x() < grid.getDimensions().x(); c.x()++ ) {
            entity.refresh();
            typename MeshType::PointType v = center< EntityType >::get( entity );
            // std::cerr << entity.getCoordinates() << " -> " << v << std::endl;
            for( int j = 0; j < v.getSize(); j++ )
               str << v[ j ] << " ";
            str << function.getData().getElement( entity.getIndex() ) << "\n";
         }
         str << "\n";
      }
      return true;
   }
};

template< typename MeshFunction, typename Real, typename Device, typename Index, int EntityDimension >
class MeshFunctionGnuplotWriter< MeshFunction, Meshes::Grid< 3, Real, Device, Index >, EntityDimension >
: public MeshFunctionGnuplotWriterBase
{
public:
   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

   static bool
   write( const MeshFunction& function, std::ostream& str )
   {
      const MeshType& grid = function.getMesh();
      EntityType entity( grid );
      auto& c = entity.getCoordinates();
      for( c.z() = 0; c.z() < grid.getDimensions().z(); c.z()++ )
         for( c.y() = 0; c.y() < grid.getDimensions().y(); c.y()++ ) {
            for( c.x() = 0; c.x() < grid.getDimensions().x(); c.x()++ ) {
               entity.refresh();
               typename MeshType::PointType v = center< EntityType >::get( entity );
               for( int j = 0; j < v.getSize(); j++ )
                  str << v[ j ] << " ";
               str << function.getData().getElement( entity.getIndex() ) << "\n";
            }
            str << "\n";
         }
      return true;
   }
};

}  // namespace Functions
}  // namespace noa::TNL
