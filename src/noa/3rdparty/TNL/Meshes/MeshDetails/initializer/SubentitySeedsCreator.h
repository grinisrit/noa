// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <noa/3rdparty/TNL/Algorithms/staticFor.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polygon.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/SubentityVertexCount.h>

namespace noaTNL {
namespace Meshes {

template< typename MeshConfig,
          typename EntityTopology,
          typename SubentityDimensionTag >
class SubentitySeedsCreator
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubentityDimensionTag::value >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;
   
public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   //using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, SubentitySeed >;
   
   /*static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray subentitySeeds;
      Algorithms::staticFor< LocalIndexType, 0, SubentitySeedArray::getSize() >(
         [&] ( auto subentityIndex ) {
            constexpr LocalIndexType subentityVerticesCount = Topologies::SubentityVertexCount< EntityTopology, SubentityTopology, subentityIndex >::count;
            auto& subentitySeed = subentitySeeds[ subentityIndex ];
            subentitySeed.setCornersCount( subentityVerticesCount );
            Algorithms::staticFor< LocalIndexType, 0, subentityVerticesCount >(
               [&] ( auto subentityVertexIndex ) {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  static constexpr LocalIndexType VERTEX_INDEX = SubentityTraits::template Vertex< decltype(subentityIndex){}, subentityVertexIndex >::index;
                  subentitySeed.setCornerId( subentityVertexIndex, subvertices.getColumnIndex( VERTEX_INDEX ) );
               }
            );
         }
      );

      return subentitySeeds;
   }*/

   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      Algorithms::staticFor< LocalIndexType, 0, SubentityTraits::count >(
         [&] ( auto subentityIndex ) {
            constexpr LocalIndexType subentityVerticesCount = Topologies::SubentityVertexCount< EntityTopology, SubentityTopology, subentityIndex >::count;
            SubentitySeed subentitySeed;
            subentitySeed.setCornersCount( subentityVerticesCount );
            Algorithms::staticFor< LocalIndexType, 0, subentityVerticesCount >(
               [&] ( auto subentityVertexIndex ) {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  static constexpr LocalIndexType VERTEX_INDEX = SubentityTraits::template Vertex< decltype(subentityIndex){}, subentityVertexIndex >::index;
                  subentitySeed.setCornerId( subentityVertexIndex, subvertices.getColumnIndex( VERTEX_INDEX ) );
               }
            );
            std::forward< FunctorType >( functor )( subentitySeed );
         }
      );
   }

   constexpr static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename MeshConfig,
          typename EntityTopology >
class SubentitySeedsCreator< MeshConfig, EntityTopology, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, SubentitySeed >;

   /*static SubentitySeedArray create( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray seeds;
      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );
      return seeds;
   }*/

   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      for( LocalIndexType i = 0; i < SubentitySeedArray::getSize(); i++ ) {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         std::forward< FunctorType >( functor )( seed );
      }
   }

   constexpr static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 1 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   
   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      for( LocalIndexType i = 0; i < subverticesCount; i++ )
      {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         seed.setCornerId( 1, subvertices.getColumnIndex( (i + 1) % subverticesCount ) );
         std::forward< FunctorType >( functor )( seed );
      }
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polygon;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;

   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      for( LocalIndexType i = 0; i < subverticesCount; i++ ) {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         std::forward< FunctorType >( functor )( seed );
      }
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 2 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType       = typename MeshTraitsType::LocalIndexType;

public:
   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      throw std::logic_error{ "Subentities of dimension 2 for polyhedrons should be initialized from seeds." };
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      throw std::logic_error{ "Subentities of dimension 2 for polyhedrons should be initialized from seeds." };
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 1 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polyhedron;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;
   using SeedSet               = typename MeshTraitsType::template EntityTraits< 1 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 1 > >;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   
   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( initializer, mesh, faceIdx, [&] ( SubentitySeed& seed ) {
            const bool inserted = seedSet.insert( seed ).second;
            if( inserted )
               std::forward< FunctorType >( functor )( seed );
         });
      }
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( initializer, mesh, faceIdx, [&] ( SubentitySeed& seed ) {
            seedSet.insert( seed );
         });
      }

      return seedSet.size();
   }
};

template< typename MeshConfig >
class SubentitySeedsCreator< MeshConfig, Topologies::Polyhedron, DimensionTag< 0 > >
{
   using MeshType              = Mesh< MeshConfig >;
   using MeshTraitsType        = MeshTraits< MeshConfig >;
   using InitializerType       = Initializer< MeshConfig >;
   using DeviceType            = typename MeshTraitsType::DeviceType;
   using GlobalIndexType       = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType        = typename MeshTraitsType::LocalIndexType;
   using EntityTopology        = Topologies::Polyhedron;
   using EntityTraitsType      = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits       = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology     = typename SubentityTraits::SubentityTopology;
   using SeedSet               = typename MeshTraitsType::template EntityTraits< 0 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< MeshConfig, Topologies::Polygon, DimensionTag< 0 > >;

public:
   using SubentitySeed = EntitySeed< MeshConfig, SubentityTopology >;
   
   template< typename FunctorType >
   static void iterate( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( initializer, mesh, faceIdx, [&] ( SubentitySeed& seed ) {
            const bool inserted = seedSet.insert( seed ).second;
            if( inserted )
               std::forward< FunctorType >( functor )( seed );
         });
      }
   }

   static LocalIndexType getSubentitiesCount( InitializerType& initializer, MeshType& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( initializer, mesh, faceIdx, [&] ( SubentitySeed& seed ) {
            seedSet.insert( seed );
         });
      }

      return seedSet.size();
   }
};

} // namespace Meshes
} // namespace noaTNL
