#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/MeshEntity.h>

namespace EntityTests {

template< typename MeshEntity >
void testVertex( const MeshEntity& entity )
{}

template< typename MeshConfig, typename Device >
void testVertex( const TNL::Meshes::MeshEntity< MeshConfig, Device, TNL::Meshes::Topologies::Vertex >& entity )
{
   EXPECT_EQ( entity.getPoint(), entity.getMesh().getPoint( entity.getIndex() ) );
}

template< int subdimension, typename MeshEntity >
void testSubentities( const MeshEntity& entity )
{
   const typename MeshEntity::MeshType mesh = entity.getMesh();
   const typename MeshEntity::GlobalIndexType index = entity.getIndex();
   constexpr int dimension = MeshEntity::getEntityDimension();

   const auto meshSubentitiesCount = mesh.template getSubentitiesCount< dimension, subdimension >( index );
   ASSERT_EQ( entity.template getSubentitiesCount< subdimension >(), meshSubentitiesCount );
   for( int i = 0; i < entity.template getSubentitiesCount< subdimension >(); i++ ) {
      const auto meshSubentityIndex = mesh.template getSubentityIndex< dimension, subdimension >( index, i );
      EXPECT_EQ( entity.template getSubentityIndex< subdimension >( i ), meshSubentityIndex );
   }
}

template< int superdimension, typename MeshEntity >
void testSuperentities( const MeshEntity& entity )
{
   const typename MeshEntity::MeshType mesh = entity.getMesh();
   const typename MeshEntity::GlobalIndexType index = entity.getIndex();
   constexpr int dimension = MeshEntity::getEntityDimension();

   const auto meshSuperentitiesCount = mesh.template getSuperentitiesCount< dimension, superdimension >( index );
   ASSERT_EQ( entity.template getSuperentitiesCount< superdimension >(), meshSuperentitiesCount );
   for( int i = 0; i < entity.template getSuperentitiesCount< superdimension >(); i++ ) {
      const auto meshSuperentityIndex = mesh.template getSuperentityIndex< dimension, superdimension >( index, i );
      EXPECT_EQ( entity.template getSuperentityIndex< superdimension >( i ), meshSuperentityIndex );
   }
}

// test if the entity is consistent with its mesh (i.e. all member functions like
// getSubentityIndex return the same value when called from the entity and the mesh)
template< typename MeshEntity >
void testEntity( const MeshEntity& entity )
{
   // static tests for the MeshEntity type
   static_assert( std::is_constructible< MeshEntity, typename MeshEntity::MeshType, typename MeshEntity::GlobalIndexType >::value,
                  "MeshEntity should be constructible from its MeshType and GlobalIndexType" );
   static_assert( ! std::is_default_constructible< MeshEntity >::value,
                  "MeshEntity should not be default-constructible" );
   static_assert( std::is_copy_constructible< MeshEntity >::value,
                  "MeshEntity should be copy-constructible" );
   static_assert( std::is_move_constructible< MeshEntity >::value,
                  "MeshEntity should be move-constructible" );
   static_assert( std::is_copy_assignable< MeshEntity >::value,
                  "MeshEntity should be copy-assignable" );
   static_assert( std::is_move_assignable< MeshEntity >::value,
                  "MeshEntity should be move-assignable" );
   static_assert( std::is_trivially_destructible< MeshEntity >::value,
                  "MeshEntity should be trivially destructible" );

   // dynamic tests for the entity
   const typename MeshEntity::MeshType mesh = entity.getMesh();
   const typename MeshEntity::GlobalIndexType index = entity.getIndex();
   constexpr int dimension = MeshEntity::getEntityDimension();

   testVertex( entity );
   EXPECT_EQ( entity.getTag(), mesh.template getEntityTag< dimension >( index ) );

   TNL::Algorithms::staticFor< int, 0, dimension >(
      [&entity] ( auto subdimension ) {
         testSubentities< subdimension >( entity );
      }
   );
   TNL::Algorithms::staticFor< int, dimension + 1, MeshEntity::MeshType::getMeshDimension() + 1 >(
      [&entity] ( auto superdimension ) {
         testSuperentities< superdimension >( entity );
      }
   );
}

template< int Dimension, typename Mesh >
void testEntities( const Mesh& mesh )
{
   using Index = typename Mesh::GlobalIndexType;
   const Index entitiesCount = mesh.template getEntitiesCount< Dimension >();
   for( Index i = 0; i < entitiesCount; i++ ) {
      const auto entity = mesh.template getEntity< Dimension >( i );
      testEntity( entity );
   }
}

} // EntityTests

template< typename Mesh >
void testEntities( const Mesh& mesh )
{
   TNL::Algorithms::staticFor< int, 0, Mesh::getMeshDimension() >(
      [&mesh] ( auto Dimension ) {
         EntityTests::testEntities< Dimension >( mesh );
      }
   );
}
#endif
