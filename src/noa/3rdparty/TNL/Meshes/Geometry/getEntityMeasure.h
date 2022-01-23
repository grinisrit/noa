// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/TNL/Meshes/GridEntity.h>
#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Meshes/MeshEntity.h>
#include <noa/3rdparty/TNL/Meshes/Geometry/getOutwardNormalVector.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Vertex.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Edge.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Triangle.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Quadrangle.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Hexahedron.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polygon.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Wedge.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Pyramid.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polyhedron.h>

namespace noa::TNL {
namespace Meshes {

template< typename Grid, typename Config >
__cuda_callable__
typename Grid::RealType
getEntityMeasure( const Grid & grid, const GridEntity< Grid, 0, Config > & entity )
{
    // entity.getMeasure() returns 0.0 !!!
    return 1.0;
}

template< typename Grid, int EntityDimension, typename Config >
__cuda_callable__
typename Grid::RealType
getEntityMeasure( const Grid & grid, const GridEntity< Grid, EntityDimension, Config > & entity )
{
    return entity.getMeasure();
}

// Vertex
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Vertex > & entity )
{
    return 1.0;
}

// Edge
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Edge > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    return l2Norm( v1 - v0 );
}

// Triangle
template< typename VectorExpression,
          std::enable_if_t< VectorExpression::getSize() == 2, bool > = true >
__cuda_callable__
typename VectorExpression::RealType
getTriangleArea( const VectorExpression & v1,
                 const VectorExpression & v2 )
{
    using Real = typename VectorExpression::RealType;
    return Real( 0.5 ) * noa::TNL::abs( v1.x() * v2.y() - v1.y() * v2.x() );
}

template< typename VectorExpression,
          std::enable_if_t< VectorExpression::getSize() == 3, bool > = true >
__cuda_callable__
typename VectorExpression::RealType
getTriangleArea( const VectorExpression & v1,
                 const VectorExpression & v2 )
{
    using Real = typename VectorExpression::RealType;
    // formula from http://math.stackexchange.com/a/128999
    const Real c1 = v1.y() * v2.z() - v1.z() * v2.y();   // first component of the cross product
    const Real c2 = v1.z() * v2.x() - v1.x() * v2.z();   // second component of the cross product
    const Real c3 = v1.x() * v2.y() - v1.y() * v2.x();   // third component of the cross product
    return Real( 0.5 ) * noa::TNL::sqrt( c1 * c1 + c2 * c2 + c3 * c3 );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Triangle > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    return getTriangleArea( v2 - v0, v1 - v0 );
}

// Quadrangle
template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Quadrangle > & entity )
{
    // measure = 0.5 * |AC x BD|, where AC and BD are the diagonals
    // Hence, we can use the same formula as for the triangle area.
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    return getTriangleArea( v2 - v0, v3 - v1 );
}

template< typename VectorExpression >
__cuda_callable__
typename VectorExpression::RealType
getTetrahedronVolume( const VectorExpression& v1,
                      const VectorExpression& v2,
                      const VectorExpression& v3 )
{
    using Real = typename VectorExpression::RealType;
    // V = (1/6) * det(v1, v2, v3)
    const Real det = v1.x() * v2.y() * v3.z() +
                     v1.y() * v2.z() * v3.x() +
                     v1.z() * v2.x() * v3.y() -
                   ( v1.z() * v2.y() * v3.x() +
                     v1.y() * v2.x() * v3.z() +
                     v1.x() * v2.z() * v3.y() );
    return Real( 1.0 / 6.0 ) * noa::TNL::abs( det );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Tetrahedron > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    return getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Hexahedron > & entity )
{
    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    const auto& v4 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 4 ) );
    const auto& v5 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 5 ) );
    const auto& v6 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 6 ) );
    const auto& v7 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 7 ) );
    // https://www.cfd-online.com/Forums/main/163122-volume-general-hexahedron.html#post574650
    return getTetrahedronVolume( v0 - v4, v3 - v4, v1 - v4 )
         + getTetrahedronVolume( v2 - v4, v3 - v4, v1 - v4 )
         + getTetrahedronVolume( v1 - v4, v2 - v4, v5 - v4 )
         + getTetrahedronVolume( v6 - v4, v2 - v4, v5 - v4 )
         + getTetrahedronVolume( v3 - v4, v2 - v4, v7 - v4 )
         + getTetrahedronVolume( v6 - v4, v2 - v4, v7 - v4 );
}

// Polygon
template< int Coord1,
          int Coord2,
          typename MeshConfig,
          typename Device >
__cuda_callable__
typename MeshConfig::RealType
getPolygon2DArea( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Polygon > & entity )
{
    // http://geomalgorithms.com/code.html (function area2D_Polygon)

    static_assert( Coord1 >= 0 && Coord1 <= 2 &&
                   Coord2 >= 0 && Coord2 <= 2 &&
                   Coord1 != Coord2, "Coord1 and Coord2 must be different integers with possible values {0, 1, 2}." );

    using Real = typename MeshConfig::RealType;
    using Index = typename MeshConfig::LocalIndexType;

    Real area{ 0.0 };
    const auto n = entity.template getSubentitiesCount< 0 >();
    for ( Index i = 1, j = 2, k = 0; j < n; i++, j++, k++ ) {
        const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( i ) );
        const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( j ) );
        const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( k ) );
        area += v0[Coord1] * ( v1[Coord2] - v2[Coord2] );
    }

    // 1. wrap around term
    {
        const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( n - 1 ) );
        const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( n - 2 ) );
        area += v0[Coord1] * ( v1[Coord2] - v2[Coord2] );
    }

    // 2. wrap around term
    {
        const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
        const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
        const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( n - 1 ) );
        area += v0[Coord1] * ( v1[Coord2] - v2[Coord2] );
    }

    return Real( 0.5 ) * area;
}

template< typename MeshConfig,
          typename Device,
          std::enable_if_t< MeshConfig::spaceDimension == 2, bool > = true >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Polygon > & entity )
{
    const auto area = getPolygon2DArea< 0, 1 >( mesh, entity );
    return noa::TNL::abs( area );
}

template< typename MeshConfig,
          typename Device,
          std::enable_if_t< MeshConfig::spaceDimension == 3, bool > = true >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Polygon > & entity )

{
    // http://geomalgorithms.com/code.html (function area3D_Polygon)

    using Real = typename MeshConfig::RealType;

    // select largest abs coordinate of normal vector to ignore for projection
    auto normal = getNormalVector( mesh, entity );
    normal = noa::TNL::abs( normal );
    int coord = 2;  // ignore z-coord
    if ( normal.x() > normal.y() ) {
        if ( normal.x() > normal.z() ) coord = 0;  // ignore x-coord
    }
    else if ( normal.y() > normal.z() ) coord = 1; // ignore y-coord

    Real area;
    switch( coord ) {
        case 0: // ignored x-coord
            area = getPolygon2DArea< 1, 2 >( mesh, entity );
            area *= l2Norm( normal ) / normal.x();
            break;
        case 1: // ignored y-coord
            area = getPolygon2DArea< 0, 2 >( mesh, entity );
            area *= l2Norm( normal ) / normal.y();
            break;
        default: // ignored z-coord
            area = getPolygon2DArea< 0, 1 >( mesh, entity );
            area *= l2Norm( normal ) / normal.z();
            break;
    }
    return noa::TNL::abs( area );
}

// Wedge
template< typename MeshConfig,
          typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Wedge > & entity )
{
    using Real = typename MeshConfig::RealType;

    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    const auto& v4 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 4 ) );
    const auto& v5 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 5 ) );
    // Partition wedge into three tetrahedrons.
    return getTetrahedronVolume( v2 - v3, v0 - v3, v1 - v3 )
         + getTetrahedronVolume( v2 - v3, v1 - v3, v4 - v3 )
         + getTetrahedronVolume( v2 - v3, v4 - v3, v5 - v3 );
}

// Pyramid
template< typename MeshConfig,
          typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Pyramid > & entity )
{
    using Real = typename MeshConfig::RealType;

    const auto& v0 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    const auto& v1 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 1 ) );
    const auto& v2 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 2 ) );
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 3 ) );
    const auto& v4 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 4 ) );
    // Partition pyramid into two tetrahedrons.
    return getTetrahedronVolume( v4 - v0, v3 - v0, v1 - v0 )
         + getTetrahedronVolume( v4 - v2, v1 - v2, v3 - v2 );
}

// Polyhedron
/*template< typename MeshConfig,
          typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Polyhedron > & entity )
{
    using Real = typename MeshConfig::RealType;
    using Index = typename MeshConfig::LocalIndexType;
    Real volume{ 0.0 };
    const Index facesCount = entity.template getSubentitiesCount< 2 >();
    for( Index faceIdx = 0; faceIdx < facesCount; faceIdx++ ) {
        const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( faceIdx ) );
        const Index verticesCount = face.template getSubentitiesCount< 0 >();
        const auto& v0 = mesh.getPoint( face.template getSubentityIndex< 0 >( 0 ) );
        for( Index i = 1, j = 2; j < verticesCount; i++, j++ ) {
            const auto& v1 = mesh.getPoint( face.template getSubentityIndex< 0 >( i ) );
            const auto& v2 = mesh.getPoint( face.template getSubentityIndex< 0 >( j ) );
            // Partition polyhedron into tetrahedrons by triangulating faces and connecting each triangle to the origin point (0,0,0).
            // It is required that vertices of all faces are stored consistently in CW or CCW order as faces are viewed from the outside.
            // Otherwise signs of some tetrahedron volumes may be incorrect, resulting in overall incorrect volume.
            // https://stackoverflow.com/a/1849746

            // volume += dot(v0 x v1, v2)
            volume += Real {
                  ( v0.y() * v1.z() - v0.z() * v1.y() ) * v2.x()
                + ( v0.z() * v1.x() - v0.x() * v1.z() ) * v2.y()
                + ( v0.x() * v1.y() - v0.y() * v1.x() ) * v2.z()
            };
        }
    }
    return Real{ 1.0 / 6.0 } * noa::TNL::abs( volume );
}*/

template< typename MeshConfig,
          typename Device >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Device > & mesh,
                  const MeshEntity< MeshConfig, Device, Topologies::Polyhedron > & entity )
{
    using Real = typename MeshConfig::RealType;
    using Index = typename MeshConfig::LocalIndexType;
    Real volume{ 0.0 };
    const Index facesCount = entity.template getSubentitiesCount< 2 >();
    const auto& v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    for( Index faceIdx = 0; faceIdx < facesCount; faceIdx++ ) {
        const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( faceIdx ) );
        const Index verticesCount = face.template getSubentitiesCount< 0 >();
        const auto& v0 = mesh.getPoint( face.template getSubentityIndex< 0 >( 0 ) );
        for( Index i = 1, j = 2; j < verticesCount; i++, j++ ) {
            const auto& v1 = mesh.getPoint( face.template getSubentityIndex< 0 >( i ) );
            const auto& v2 = mesh.getPoint( face.template getSubentityIndex< 0 >( j ) );
            // Partition polyhedron into tetrahedrons by triangulating faces and connecting each triangle to one point of the polyhedron.
            volume += getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
        }
    }
    return volume;
}

template< typename MeshConfig >
__cuda_callable__
typename MeshConfig::RealType
getEntityMeasure( const Mesh< MeshConfig, Devices::Cuda > & mesh,
                  const MeshEntity< MeshConfig, Devices::Cuda, Topologies::Polyhedron > & entity )
{
    using Real = typename MeshConfig::RealType;
    using Index = typename MeshConfig::LocalIndexType;
    using Point = typename Mesh< MeshConfig, Devices::Cuda >::PointType;
    Real volume{ 0.0 };
    const Index facesCount = entity.template getSubentitiesCount< 2 >();
    const Point v3 = mesh.getPoint( entity.template getSubentityIndex< 0 >( 0 ) );
    for( Index faceIdx = 0; faceIdx < facesCount; faceIdx++ ) {
        const auto face = mesh.template getEntity< 2 >( entity.template getSubentityIndex< 2 >( faceIdx ) );
        const Index verticesCount = face.template getSubentitiesCount< 0 >();
        const Point v0 = mesh.getPoint( face.template getSubentityIndex< 0 >( 0 ) );
        Point v1 = mesh.getPoint( face.template getSubentityIndex< 0 >( 1 ) );
        for( Index j = 2; j < verticesCount; j++ ) {
            const Point v2 = mesh.getPoint( face.template getSubentityIndex< 0 >( j ) );
            // Partition polyhedron into tetrahedrons by triangulating faces and connecting each triangle to one point of the polyhedron.
            volume += getTetrahedronVolume( v3 - v0, v2 - v0, v1 - v0 );
            v1 = v2;
        }
    }
    return volume;
}

} // namespace Meshes
} // namespace noa::TNL
