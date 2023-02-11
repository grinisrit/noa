// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {
namespace detail {

// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension >
struct MeshEntitiesVTUCollector
{
   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         const Index verticesPerEntity = entity.template getSubentitiesCount< 0 >();
         for( Index j = 0; j < verticesPerEntity; j++ )
            connectivity.push_back( entity.template getSubentityIndex< 0 >( j ) );
         offsets.push_back( connectivity.size() );
         const auto type =
            static_cast< std::uint8_t >( VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape );
         types.push_back( type );
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct MeshEntitiesVTUCollector< Mesh, 0 >
{
   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         connectivity.push_back( i );
         offsets.push_back( connectivity.size() );
         const auto type =
            static_cast< std::uint8_t >( VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape );
         types.push_back( type );
      }
   }
};

// 1D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 1 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
         connectivity.push_back( i );
         connectivity.push_back( i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
      }
   }
};

// 1D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 0 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ ) {
         connectivity.push_back( i );
         offsets.push_back( connectivity.size() );
         types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
      }
   }
};

// 2D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 2 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
            connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            connectivity.push_back( ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
            connectivity.push_back( ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            offsets.push_back( connectivity.size() );
            types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
         }
   }
};

// 2D grids, faces
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 1 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
            connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
            connectivity.push_back( ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
            offsets.push_back( connectivity.size() );
            types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
         }

      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
            connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            offsets.push_back( connectivity.size() );
            types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
         }
   }
};

// 2D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 0 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
            connectivity.push_back( j * mesh.getDimensions().x() + i );
            offsets.push_back( connectivity.size() );
            types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
         }
   }
};

// 3D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 3 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }
   }
};

// 3D grids, faces
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 2 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }
   }
};

// 3D grids, edges
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 1 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               connectivity.push_back( ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }
   }
};

// 3D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using Entity = typename Mesh::template EntityType< 0 >;

   static void
   exec( const Mesh& mesh,
         std::vector< typename Mesh::GlobalIndexType >& connectivity,
         std::vector< typename Mesh::GlobalIndexType >& offsets,
         std::vector< std::uint8_t >& types )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
         for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
            for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
               connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               offsets.push_back( connectivity.size() );
               types.push_back( static_cast< std::uint8_t >( VTK::GridEntityShape< Entity >::shape ) );
            }
   }
};

}  // namespace detail
}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
