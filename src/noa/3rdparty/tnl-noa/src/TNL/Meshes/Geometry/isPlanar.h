// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityMeasure.h>

namespace noa::TNL {
namespace Meshes {

// Polygon
template< typename MeshConfig, typename Device, std::enable_if_t< MeshConfig::spaceDimension == 3, bool > = true >
__cuda_callable__
bool
isPlanar( const Mesh< MeshConfig, Device >& mesh,
          const MeshEntity< MeshConfig, Device, Topologies::Polygon >& entity,
          const typename MeshConfig::RealType precision )
{
   using Real = typename MeshConfig::RealType;
   using Index = typename MeshConfig::LocalIndexType;
   const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
   const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
   const Index verticesCount = entity.template getSubentitiesCount< 0 >();
   for( Index i = 2, j = 3; j < verticesCount; i++, j++ ) {
      const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( i ) );
      const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( j ) );
      const Real volume{ getTetrahedronVolume( v0 - v1, v2 - v1, v3 - v1 ) };
      if( volume > precision )
         return false;
   }
   return true;
}

}  // namespace Meshes
}  // namespace noa::TNL
