// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/FPMAWriter.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

namespace details {

inline void
writeInt( std::ostream& str, std::int32_t value )
{
   str << value << ' ';
}

template< typename Real >
void
writeReal( std::ostream& str, const Real value )
{
   str.precision( std::numeric_limits< Real >::digits10 );
   str << value << ' ';
}

template< typename Mesh, int EntityDimension, int SubDimension >
struct MeshEntitiesFPMAWriter
{
   static void
   exec( const Mesh& mesh, std::ostream& str )
   {
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      str << '\n' << entitiesCount;
      for( Index i = 0; i < entitiesCount; i++ ) {
         str << '\n';
         const auto& entity = mesh.template getEntity< EntityDimension >( i );
         const Index subentitiesPerEntity = entity.template getSubentitiesCount< SubDimension >();
         writeInt( str, subentitiesPerEntity );
         for( Index j = 0; j < subentitiesPerEntity; j++ )
            writeInt( str, entity.template getSubentityIndex< SubDimension >( j ) );
      }
   }
};

}  // namespace details

template< typename Mesh >
void
FPMAWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   writePoints( mesh );
   EntitiesWriter< 2, 0 >::exec( mesh, str );
   EntitiesWriter< 3, 2 >::exec( mesh, str );
}

template< typename Mesh >
void
FPMAWriter< Mesh >::writePoints( const Mesh& mesh )
{
   IndexType pointsCount = mesh.template getEntitiesCount< 0 >();
   str << pointsCount << '\n';
   for( IndexType i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< 0 >( i );
      const auto& point = vertex.getPoint();
      for( IndexType j = 0; j < point.getSize(); j++ )
         details::writeReal( str, point[ j ] );
   }
}

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
