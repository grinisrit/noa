#pragma once

#include <type_traits>

template< typename Mesh, typename EntityType >
typename Mesh::GlobalIndexType
mesh_getEntitiesCount( const Mesh & self, const EntityType & entity )
{
    static_assert( std::is_same< EntityType, typename Mesh::Cell >::value ||
                   std::is_same< EntityType, typename Mesh::Face >::value ||
                   std::is_same< EntityType, typename Mesh::Vertex >::value,
                   "incompatible entity type" );
    return self.template getEntitiesCount< EntityType::getEntityDimension() >();
}

template< typename Mesh, typename EntityType >
typename Mesh::GlobalIndexType
mesh_getGhostEntitiesCount( const Mesh & self, const EntityType & entity )
{
    static_assert( std::is_same< EntityType, typename Mesh::Cell >::value ||
                   std::is_same< EntityType, typename Mesh::Face >::value ||
                   std::is_same< EntityType, typename Mesh::Vertex >::value,
                   "incompatible entity type" );
    return self.template getGhostEntitiesCount< EntityType::getEntityDimension() >();
}

template< typename Mesh, typename EntityType >
typename Mesh::GlobalIndexType
mesh_getGhostEntitiesOffset( const Mesh & self, const EntityType & entity )
{
    static_assert( std::is_same< EntityType, typename Mesh::Cell >::value ||
                   std::is_same< EntityType, typename Mesh::Face >::value ||
                   std::is_same< EntityType, typename Mesh::Vertex >::value,
                   "incompatible entity type" );
    return self.template getGhostEntitiesOffset< EntityType::getEntityDimension() >();
}
