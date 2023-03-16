// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/staticFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntitySeed.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polygon.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexCount.h>

namespace noa::TNL {
namespace Meshes {

template< typename Mesh, typename EntityTopology, typename SubentityDimensionTag >
class SubentitySeedsCreator
{
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits =
      typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, SubentityDimensionTag::value >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;
   // using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, SubentitySeed >;

   /*static SubentitySeedArray create( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray subentitySeeds;
      Algorithms::staticFor< LocalIndexType, 0, SubentitySeedArray::getSize() >(
         [&] ( auto subentityIndex ) {
            constexpr LocalIndexType subentityVerticesCount = Topologies::SubentityVertexCount< EntityTopology,
   SubentityTopology, subentityIndex >::count; auto& subentitySeed = subentitySeeds[ subentityIndex ];
            subentitySeed.setCornersCount( subentityVerticesCount );
            Algorithms::staticFor< LocalIndexType, 0, subentityVerticesCount >(
               [&] ( auto subentityVertexIndex ) {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  constexpr LocalIndexType vertexIndex = SubentityTraits::template Vertex< decltype(subentityIndex){},
   subentityVertexIndex >::index; subentitySeed.setCornerId( subentityVertexIndex, subvertices.getColumnIndex( vertexIndex ) );
               }
            );
         }
      );

      return subentitySeeds;
   }*/

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

// FIXME: nvcc 11.8 fails to compile nested staticFor loops with different index types (note that the outer loop is in
// EntityInitializer.h and it uses `int` to iterate over dimensions)
#ifdef __NVCC__
      using LocalIndexType = int;
#endif

      Algorithms::staticFor< LocalIndexType, 0, SubentityTraits::count >(
         [ & ]( auto subentityIndex )
         {
            constexpr LocalIndexType subentityVerticesCount =
               Topologies::SubentityVertexCount< EntityTopology, SubentityTopology, subentityIndex >::count;
            SubentitySeed subentitySeed;
            subentitySeed.setCornersCount( subentityVerticesCount );
            Algorithms::staticFor< LocalIndexType, 0, subentityVerticesCount >(
               [ & ]( auto subentityVertexIndex )
               {
                  // subentityIndex cannot be captured as constexpr, so we need to create another instance of its type
                  constexpr LocalIndexType vertexIndex =
                     SubentityTraits::template Vertex< decltype( subentityIndex ){}, subentityVertexIndex >::index;
                  subentitySeed.setCornerId( subentityVertexIndex, subvertices.getColumnIndex( vertexIndex ) );
               } );
            functor( subentitySeed );
         } );
   }

   constexpr static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename Mesh, typename EntityTopology >
class SubentitySeedsCreator< Mesh, EntityTopology, DimensionTag< 0 > >
{
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits = typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;
   using SubentitySeedArray = Containers::StaticArray< SubentityTraits::count, SubentitySeed >;

   /*static SubentitySeedArray create( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      SubentitySeedArray seeds;
      for( LocalIndexType i = 0; i < seeds.getSize(); i++ )
         seeds[ i ].setCornerId( 0, subvertices.getColumnIndex( i ) );
      return seeds;
   }*/

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );

      for( LocalIndexType i = 0; i < SubentitySeedArray::getSize(); i++ ) {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         functor( seed );
      }
   }

   constexpr static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      return SubentityTraits::count;
   }
};

template< typename Mesh >
class SubentitySeedsCreator< Mesh, Topologies::Polygon, DimensionTag< 1 > >
{
   using DeviceType = typename Mesh::DeviceType;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTopology = Topologies::Polygon;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits = typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      for( LocalIndexType i = 0; i < subverticesCount; i++ ) {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         seed.setCornerId( 1, subvertices.getColumnIndex( ( i + 1 ) % subverticesCount ) );
         functor( seed );
      }
   }

   static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

template< typename Mesh >
class SubentitySeedsCreator< Mesh, Topologies::Polygon, DimensionTag< 0 > >
{
   using DeviceType = typename Mesh::DeviceType;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTopology = Topologies::Polygon;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits = typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      const auto& subvertices = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 0 >().getRow( entityIndex );
      const LocalIndexType subverticesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );

      for( LocalIndexType i = 0; i < subverticesCount; i++ ) {
         SubentitySeed seed;
         seed.setCornerId( 0, subvertices.getColumnIndex( i ) );
         functor( seed );
      }
   }

   static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      return mesh.template getSubentitiesCount< EntityTopology::dimension, 0 >( entityIndex );
   }
};

template< typename Mesh >
class SubentitySeedsCreator< Mesh, Topologies::Polyhedron, DimensionTag< 2 > >
{
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;

public:
   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      throw std::logic_error{ "Subentities of dimension 2 for polyhedrons should be initialized from seeds." };
   }

   static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      throw std::logic_error{ "Subentities of dimension 2 for polyhedrons should be initialized from seeds." };
   }
};

template< typename Mesh >
class SubentitySeedsCreator< Mesh, Topologies::Polyhedron, DimensionTag< 1 > >
{
   using DeviceType = typename Mesh::DeviceType;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTopology = Topologies::Polyhedron;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits = typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, 1 >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;
   using SeedSet = typename Mesh::MeshTraitsType::template EntityTraits< 1 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< Mesh, Topologies::Polygon, DimensionTag< 1 > >;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( mesh,
                                             faceIdx,
                                             [ & ]( SubentitySeed& seed )
                                             {
                                                const bool inserted = seedSet.insert( seed ).second;
                                                if( inserted )
                                                   functor( seed );
                                             } );
      }
   }

   static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );
      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( mesh,
                                             faceIdx,
                                             [ & ]( SubentitySeed& seed )
                                             {
                                                seedSet.insert( seed );
                                             } );
      }

      return seedSet.size();
   }
};

template< typename Mesh >
class SubentitySeedsCreator< Mesh, Topologies::Polyhedron, DimensionTag< 0 > >
{
   using DeviceType = typename Mesh::DeviceType;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;
   using EntityTopology = Topologies::Polyhedron;
   using EntityTraitsType = typename Mesh::MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using SubentityTraits = typename Mesh::MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;
   using SubentityTopology = typename SubentityTraits::SubentityTopology;
   using SeedSet = typename Mesh::MeshTraitsType::template EntityTraits< 0 >::SeedSetType;
   using FaceSubentitySeedsCreator = SubentitySeedsCreator< Mesh, Topologies::Polygon, DimensionTag< 0 > >;

public:
   using SubentitySeed = EntitySeed< typename Mesh::Config, SubentityTopology >;

   template< typename FunctorType >
   static void
   iterate( Mesh& mesh, const GlobalIndexType entityIndex, FunctorType&& functor )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( mesh,
                                             faceIdx,
                                             [ & ]( SubentitySeed& seed )
                                             {
                                                const bool inserted = seedSet.insert( seed ).second;
                                                if( inserted )
                                                   functor( seed );
                                             } );
      }
   }

   static LocalIndexType
   getSubentitiesCount( Mesh& mesh, const GlobalIndexType entityIndex )
   {
      SeedSet seedSet;
      const auto& faces = mesh.template getSubentitiesMatrix< EntityTopology::dimension, 2 >().getRow( entityIndex );
      const LocalIndexType facesCount = mesh.template getSubentitiesCount< EntityTopology::dimension, 2 >( entityIndex );

      for( LocalIndexType i = 0; i < facesCount; i++ ) {
         GlobalIndexType faceIdx = faces.getColumnIndex( i );
         FaceSubentitySeedsCreator::iterate( mesh,
                                             faceIdx,
                                             [ & ]( SubentitySeed& seed )
                                             {
                                                seedSet.insert( seed );
                                             } );
      }

      return seedSet.size();
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
