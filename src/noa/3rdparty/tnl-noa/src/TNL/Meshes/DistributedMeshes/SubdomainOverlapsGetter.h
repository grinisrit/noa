// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename Mesh >
class DistributedMesh;

template< typename Mesh >
class SubdomainOverlapsGetter
{};

// TODO: Specializations by the grid dimension can be avoided when the MPI directions are
// rewritten in a dimension independent way

template< typename Real, typename Device, typename Index >
class SubdomainOverlapsGetter< Grid< 1, Real, Device, Index > >
{
public:
   static const int Dimension = 1;
   using MeshType = Grid< Dimension, Real, Device, Index >;
   using DeviceType = Device;
   using IndexType = Index;
   using DistributedMeshType = DistributedMesh< MeshType >;
   using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
   using CoordinatesType = typename DistributedMeshType::CoordinatesType;

   // Computes subdomain overlaps
   /* SubdomainOverlapsType is a touple of the same size as the mesh dimensions.
    * lower.x() is overlap of the subdomain at boundary where x = 0,
    * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
    */
   static void
   getOverlaps( const DistributedMeshType* distributedMesh,
                SubdomainOverlapsType& lower,
                SubdomainOverlapsType& upper,
                IndexType subdomainOverlapSize,
                const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize = 0,
                const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize = 0 );
};

template< typename Real, typename Device, typename Index >
class SubdomainOverlapsGetter< Grid< 2, Real, Device, Index > >
{
public:
   static const int Dimension = 2;
   using MeshType = Grid< Dimension, Real, Device, Index >;
   using DeviceType = Device;
   using IndexType = Index;
   using DistributedMeshType = DistributedMesh< MeshType >;
   using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
   using CoordinatesType = typename DistributedMeshType::CoordinatesType;

   // Computes subdomain overlaps
   /* SubdomainOverlapsType is a touple of the same size as the mesh dimensions.
    * lower.x() is overlap of the subdomain at boundary where x = 0,
    * lower.y() is overlap of the subdomain at boundary where y = 0,
    * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
    * upper.y() is overlap of the subdomain at boundary where y = grid.getDimensions().y() - 1.
    */
   static void
   getOverlaps( const DistributedMeshType* distributedMesh,
                SubdomainOverlapsType& lower,
                SubdomainOverlapsType& upper,
                IndexType subdomainOverlapSize,
                const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize = 0,
                const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize = 0 );
};

template< typename Real, typename Device, typename Index >
class SubdomainOverlapsGetter< Grid< 3, Real, Device, Index > >
{
public:
   static const int Dimension = 3;
   using MeshType = Grid< Dimension, Real, Device, Index >;
   using DeviceType = Device;
   using IndexType = Index;
   using DistributedMeshType = DistributedMesh< MeshType >;
   using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
   using CoordinatesType = typename DistributedMeshType::CoordinatesType;

   // Computes subdomain overlaps
   /* SubdomainOverlapsType is a touple of the same size as the mesh dimensions.
    * lower.x() is overlap of the subdomain at boundary where x = 0,
    * lower.y() is overlap of the subdomain at boundary where y = 0,
    * lower.z() is overlap of the subdomain at boundary where z = 0,
    * upper.x() is overlap of the subdomain at boundary where x = grid.getDimensions().x() - 1,
    * upper.y() is overlap of the subdomain at boundary where y = grid.getDimensions().y() - 1,
    * upper.z() is overlap of the subdomain at boundary where z = grid.getDimensions().z() - 1,
    */
   static void
   getOverlaps( const DistributedMeshType* distributedMesh,
                SubdomainOverlapsType& lower,
                SubdomainOverlapsType& upper,
                IndexType subdomainOverlapSize,
                const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize = 0,
                const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize = 0 );
};

}  // namespace DistributedMeshes
}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.hpp>
