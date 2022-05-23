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

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntitySeedMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Ellpack.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SlicedEllpack.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Vertex.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polyhedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/IsDynamicTopology.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology >
class MeshEntity;

template< typename MeshConfig,
          typename EntityTopology,
          bool IsDynamicTopology = Topologies::IsDynamicTopology< EntityTopology >::value >
class EntitySeed;

template< typename MeshConfig, typename DimensionTag >
struct EntityTopologyGetter
{
   static_assert( DimensionTag::value <= MeshConfig::meshDimension,
                  "There are no entities with dimension higher than the mesh dimension." );
   using Topology = typename Topologies::Subtopology< typename MeshConfig::CellTopology, DimensionTag::value >::Topology;
};

template< typename MeshConfig >
struct EntityTopologyGetter< MeshConfig, DimensionTag< MeshConfig::CellTopology::dimension > >
{
   using Topology = typename MeshConfig::CellTopology;
};

template< typename MeshConfig,
          typename Device,
          int Dimension,
          bool IsDynamicTopology = Topologies::IsDynamicTopology<
             typename EntityTopologyGetter< MeshConfig, DimensionTag< Dimension > >::Topology >::value >
class MeshEntityTraits;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          int Dimension,
          bool IsDynamicTopology = Topologies::IsDynamicTopology< EntityTopology >::value >
class MeshSubentityTraits;

template< typename MeshConfig, typename Device, typename MeshEntity, int Superdimension >
class MeshSuperentityTraits;

// helper templates (must be public because nvcc sucks, and outside of MeshTraits to avoid duplicate code generation)
template< typename Device, typename Index, typename IndexAlocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAlocator >;
template< typename Device, typename Index, typename IndexAlocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAlocator >;

template< typename MeshConfig, typename Device = Devices::Host >
class MeshTraits
{
public:
   static constexpr int meshDimension = MeshConfig::CellTopology::dimension;
   static constexpr int spaceDimension = MeshConfig::spaceDimension;

   using DeviceType = Device;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;

   using CellTopology = typename MeshConfig::CellTopology;
   using FaceTopology = typename Topologies::Subtopology< CellTopology, meshDimension - 1 >::Topology;
   using CellType = MeshEntity< MeshConfig, Device, CellTopology >;
   using VertexType = MeshEntity< MeshConfig, Device, Topologies::Vertex >;
   using PointType = Containers::StaticVector< spaceDimension, typename MeshConfig::RealType >;
   using FaceSeedType = EntitySeed< MeshConfig, FaceTopology >;
   using CellSeedType = EntitySeed< MeshConfig, CellTopology >;
   using EntityTagType = std::uint8_t;

   using NeighborCountsArray = Containers::Vector< LocalIndexType, DeviceType, GlobalIndexType >;
   using PointArrayType = Containers::Array< PointType, DeviceType, GlobalIndexType >;
   using FaceSeedMatrixType = EntitySeedMatrix< MeshConfig, FaceTopology >;
   using CellSeedMatrixType = EntitySeedMatrix< MeshConfig, CellTopology >;

   using EntityTagsArrayType = Containers::Array< EntityTagType, DeviceType, GlobalIndexType >;

   template< int Dimension >
   using EntityTraits = MeshEntityTraits< MeshConfig, DeviceType, Dimension >;

   template< typename EntityTopology, int Subdimension >
   using SubentityTraits = MeshSubentityTraits< MeshConfig, DeviceType, EntityTopology, Subdimension >;

   template< typename EntityTopology, int Superdimension >
   using SuperentityTraits = MeshSuperentityTraits< MeshConfig, DeviceType, EntityTopology, Superdimension >;

   using DimensionTag = Meshes::DimensionTag< meshDimension >;

   // container for storing the subentity indices
   template< int Dimension >
   using SubentityMatrixType = typename EntityTraits< Dimension >::SubentityMatrixType;

   // container for storing the superentity indices
   using SuperentityMatrixType =
      Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >;

   // container for storing the dual graph adjacency matrix
   using DualGraph = Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >;
};

}  // namespace Meshes
}  // namespace noa::TNL
