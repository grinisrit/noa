// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Triangle.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Quadrangle.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Hexahedron.h>

namespace noa::TNL {
namespace Meshes {

enum class EntityRefinerVersion
{
   EdgeBisection
};

template< typename MeshConfig,
          typename Topology,
          EntityRefinerVersion EntityRefinerVersion_ = EntityRefinerVersion::EdgeBisection >
struct EntityRefiner;

template< typename MeshConfig >
struct EntityRefiner< MeshConfig, Topologies::Triangle, EntityRefinerVersion::EdgeBisection >
{
   using Device = Devices::Host;
   using Topology = Topologies::Triangle;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   // returns: number of *new* points, number of *all* refined entities
   static std::pair< GlobalIndexType, GlobalIndexType >
   getExtraPointsAndEntitiesCount( const MeshEntityType& entity )
   {
      return { 3, 4 };
   }

   template< typename AddPointFunctor, typename AddCellFunctor >
   static void
   decompose( const MeshEntityType& entity, AddPointFunctor&& addPoint, AddCellFunctor&& addCell )
   {
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      const auto v1 = entity.template getSubentityIndex< 0 >( 1 );
      const auto v2 = entity.template getSubentityIndex< 0 >( 2 );

      const auto& mesh = entity.getMesh();
      const auto v0_p = mesh.getPoint( v0 );
      const auto v1_p = mesh.getPoint( v1 );
      const auto v2_p = mesh.getPoint( v2 );

      // add new points: midpoints of triangle edges
      const auto w0 = addPoint( 0.5 * ( v1_p + v2_p ) );
      const auto w1 = addPoint( 0.5 * ( v0_p + v2_p ) );
      const auto w2 = addPoint( 0.5 * ( v0_p + v1_p ) );

      // add refined triangles
      addCell( v0, w1, w2 );
      addCell( v1, w0, w2 );
      addCell( v2, w0, w1 );
      addCell( w0, w1, w2 );
   }
};

template< typename MeshConfig >
struct EntityRefiner< MeshConfig, Topologies::Quadrangle, EntityRefinerVersion::EdgeBisection >
{
   using Device = Devices::Host;
   using Topology = Topologies::Quadrangle;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   // returns: number of *new* points, number of *all* refined entities
   static std::pair< GlobalIndexType, GlobalIndexType >
   getExtraPointsAndEntitiesCount( const MeshEntityType& entity )
   {
      return { 5, 4 };
   }

   template< typename AddPointFunctor, typename AddCellFunctor >
   static void
   decompose( const MeshEntityType& entity, AddPointFunctor&& addPoint, AddCellFunctor&& addCell )
   {
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      const auto v1 = entity.template getSubentityIndex< 0 >( 1 );
      const auto v2 = entity.template getSubentityIndex< 0 >( 2 );
      const auto v3 = entity.template getSubentityIndex< 0 >( 3 );

      const auto& mesh = entity.getMesh();
      const auto v0_p = mesh.getPoint( v0 );
      const auto v1_p = mesh.getPoint( v1 );
      const auto v2_p = mesh.getPoint( v2 );
      const auto v3_p = mesh.getPoint( v3 );

      // add new points
      const auto w0 = addPoint( 0.5 * ( v0_p + v1_p ) );
      const auto w1 = addPoint( 0.5 * ( v1_p + v2_p ) );
      const auto w2 = addPoint( 0.5 * ( v2_p + v3_p ) );
      const auto w3 = addPoint( 0.5 * ( v3_p + v0_p ) );
      const auto c = addPoint( 0.25 * ( v0_p + v1_p + v2_p + v3_p ) );

      // add refined quadrangles
      addCell( v0, w0, c, w3 );
      addCell( w0, v1, w1, c );
      addCell( c, w1, v2, w2 );
      addCell( w3, c, w2, v3 );
   }
};

template< typename MeshConfig >
struct EntityRefiner< MeshConfig, Topologies::Tetrahedron, EntityRefinerVersion::EdgeBisection >
{
   using Device = Devices::Host;
   using Topology = Topologies::Tetrahedron;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   // returns: number of *new* points, number of *all* refined entities
   static std::pair< GlobalIndexType, GlobalIndexType >
   getExtraPointsAndEntitiesCount( const MeshEntityType& entity )
   {
      return { 6, 8 };
   }

   template< typename AddPointFunctor, typename AddCellFunctor >
   static void
   decompose( const MeshEntityType& entity, AddPointFunctor&& addPoint, AddCellFunctor&& addCell )
   {
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      const auto v1 = entity.template getSubentityIndex< 0 >( 1 );
      const auto v2 = entity.template getSubentityIndex< 0 >( 2 );
      const auto v3 = entity.template getSubentityIndex< 0 >( 3 );

      const auto& mesh = entity.getMesh();
      const auto v0_p = mesh.getPoint( v0 );
      const auto v1_p = mesh.getPoint( v1 );
      const auto v2_p = mesh.getPoint( v2 );
      const auto v3_p = mesh.getPoint( v3 );

      // add new points: midpoints of triangle edges
      const auto w0 = addPoint( 0.5 * ( v1_p + v2_p ) );
      const auto w1 = addPoint( 0.5 * ( v0_p + v2_p ) );
      const auto w2 = addPoint( 0.5 * ( v0_p + v1_p ) );
      const auto w3 = addPoint( 0.5 * ( v0_p + v3_p ) );
      const auto w4 = addPoint( 0.5 * ( v1_p + v3_p ) );
      const auto w5 = addPoint( 0.5 * ( v2_p + v3_p ) );

      // add refined tetrahedrons
      addCell( v0, w1, w2, w3 );
      addCell( v1, w0, w2, w4 );
      addCell( v2, w0, w1, w5 );
      addCell( v3, w3, w4, w5 );

      addCell( w5, w0, w1, w2 );
      addCell( w5, w1, w2, w3 );
      addCell( w5, w0, w2, w4 );
      addCell( w5, w3, w4, w2 );
   }
};

template< typename MeshConfig >
struct EntityRefiner< MeshConfig, Topologies::Hexahedron, EntityRefinerVersion::EdgeBisection >
{
   using Device = Devices::Host;
   using Topology = Topologies::Hexahedron;
   using MeshEntityType = MeshEntity< MeshConfig, Device, Topology >;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   // returns: number of *new* points, number of *all* refined entities
   static std::pair< GlobalIndexType, GlobalIndexType >
   getExtraPointsAndEntitiesCount( const MeshEntityType& entity )
   {
      return { 19, 8 };
   }

   template< typename AddPointFunctor, typename AddCellFunctor >
   static void
   decompose( const MeshEntityType& entity, AddPointFunctor&& addPoint, AddCellFunctor&& addCell )
   {
      const auto v0 = entity.template getSubentityIndex< 0 >( 0 );
      const auto v1 = entity.template getSubentityIndex< 0 >( 1 );
      const auto v2 = entity.template getSubentityIndex< 0 >( 2 );
      const auto v3 = entity.template getSubentityIndex< 0 >( 3 );
      const auto v4 = entity.template getSubentityIndex< 0 >( 4 );
      const auto v5 = entity.template getSubentityIndex< 0 >( 5 );
      const auto v6 = entity.template getSubentityIndex< 0 >( 6 );
      const auto v7 = entity.template getSubentityIndex< 0 >( 7 );

      const auto& mesh = entity.getMesh();
      const auto v0_p = mesh.getPoint( v0 );
      const auto v1_p = mesh.getPoint( v1 );
      const auto v2_p = mesh.getPoint( v2 );
      const auto v3_p = mesh.getPoint( v3 );
      const auto v4_p = mesh.getPoint( v4 );
      const auto v5_p = mesh.getPoint( v5 );
      const auto v6_p = mesh.getPoint( v6 );
      const auto v7_p = mesh.getPoint( v7 );

      // add new points: centers of bottom edges
      const auto b0 = addPoint( 0.5 * ( v0_p + v1_p ) );
      const auto b1 = addPoint( 0.5 * ( v1_p + v2_p ) );
      const auto b2 = addPoint( 0.5 * ( v2_p + v3_p ) );
      const auto b3 = addPoint( 0.5 * ( v3_p + v0_p ) );
      // add new points: centers of upper edges
      const auto u0 = addPoint( 0.5 * ( v4_p + v5_p ) );
      const auto u1 = addPoint( 0.5 * ( v5_p + v6_p ) );
      const auto u2 = addPoint( 0.5 * ( v6_p + v7_p ) );
      const auto u3 = addPoint( 0.5 * ( v7_p + v4_p ) );
      // add new points: centers of middle (vertical) edges
      const auto m0 = addPoint( 0.5 * ( v0_p + v4_p ) );
      const auto m1 = addPoint( 0.5 * ( v1_p + v5_p ) );
      const auto m2 = addPoint( 0.5 * ( v2_p + v6_p ) );
      const auto m3 = addPoint( 0.5 * ( v3_p + v7_p ) );
      // add new points: centers of faces
      const auto f0 = addPoint( 0.25 * ( v0_p + v1_p + v2_p + v3_p ) );
      const auto f1 = addPoint( 0.25 * ( v0_p + v1_p + v5_p + v4_p ) );
      const auto f2 = addPoint( 0.25 * ( v1_p + v2_p + v6_p + v5_p ) );
      const auto f3 = addPoint( 0.25 * ( v2_p + v3_p + v7_p + v6_p ) );
      const auto f4 = addPoint( 0.25 * ( v3_p + v0_p + v4_p + v7_p ) );
      const auto f5 = addPoint( 0.25 * ( v4_p + v5_p + v6_p + v7_p ) );
      // add new points: center of the cell
      const auto cc = addPoint( 0.125 * ( v0_p + v1_p + v2_p + v3_p + v4_p + v5_p + v6_p + v7_p ) );

      // add refined hexahedrons
      addCell( v0, b0, f0, b3, m0, f1, cc, f4 );
      addCell( b0, v1, b1, f0, f1, m1, f2, cc );
      addCell( f0, b1, v2, b2, cc, f2, m2, f3 );
      addCell( b3, f0, b2, v3, f4, cc, f3, m3 );
      addCell( m0, f1, cc, f4, v4, u0, f5, u3 );
      addCell( f1, m1, f2, cc, u0, v5, u1, f5 );
      addCell( cc, f2, m2, f3, f5, u1, v6, u2 );
      addCell( f4, cc, f3, m3, u3, f5, u2, v7 );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
