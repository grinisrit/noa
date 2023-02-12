// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityMeasure.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename Topology >
__cuda_callable__
typename MeshConfig::RealType
getEntityCircumradius( const Mesh< MeshConfig, Device >& mesh, const MeshEntity< MeshConfig, Device, Topology >& entity )
{
   return std::numeric_limits< typename MeshConfig::RealType >::quiet_NaN();
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityCircumradius( const Mesh< MeshConfig, Device >& mesh,
                       const MeshEntity< MeshConfig, Device, Topologies::Edge >& entity )
{
   using Real = typename MeshConfig::RealType;
   return Real( 0.5 ) * getEntityMeasure( mesh, entity );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityCircumradius( const Mesh< MeshConfig, Device >& mesh,
                       const MeshEntity< MeshConfig, Device, Topologies::Triangle >& entity )
{
   const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
   const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
   const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );

   const auto a = TNL::l2Norm( v1 - v2 );
   const auto b = TNL::l2Norm( v1 - v0 );
   const auto c = TNL::l2Norm( v2 - v0 );

   // https://en.wikipedia.org/wiki/Circumradius
   // getTriangleArea returns half of the cross product
   return a * b * c / ( 2 * 2 * getTriangleArea( v1 - v0, v2 - v0 ) );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityCircumradius( const Mesh< MeshConfig, Device >& mesh,
                       const MeshEntity< MeshConfig, Device, Topologies::Tetrahedron >& entity )
{
   const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
   const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
   const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
   const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );

   // https://en.wikipedia.org/wiki/Tetrahedron#Circumradius
   const auto a = TNL::l2Norm( v1 - v0 );
   const auto b = TNL::l2Norm( v2 - v0 );
   const auto c = TNL::l2Norm( v3 - v0 );
   const auto A = TNL::l2Norm( v3 - v2 );  // opposite to v1-v0
   const auto B = TNL::l2Norm( v3 - v1 );  // opposite to v2-v0
   const auto C = TNL::l2Norm( v2 - v1 );  // opposite to v3-v0
   const auto V = getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
   const auto product =
      ( a * A + b * B + c * C ) * ( a * A + b * B - c * C ) * ( a * A - b * B + c * C ) * ( -a * A + b * B + c * C );
   return TNL::sqrt( product ) / ( 24 * V );
}

}  // namespace Meshes
}  // namespace noa::TNL
