// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Edge.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Triangle.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Quadrangle.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Hexahedron.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Simplex.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Wedge.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Pyramid.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polyhedron.h>

namespace noa::TNL {
namespace Meshes {
/**
 * \brief Namespace for the configuration of the \ref GridTypeResolver and
 * \ref MeshTypeResolver using so-called build config tags and partial class
 * template specializations.
 */
namespace BuildConfigTags {

// Configuration for structured grids

// 1, 2, and 3 dimensions are enabled by default
template< typename ConfigTag, int Dimension > struct GridDimensionTag { static constexpr bool enabled = Dimension > 0 && Dimension <= 3; };

// Grids are enabled only for the `float` and `double` real types by default.
template< typename ConfigTag, typename Real > struct GridRealTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct GridRealTag< ConfigTag, float > { static constexpr bool enabled = true; };
template< typename ConfigTag > struct GridRealTag< ConfigTag, double > { static constexpr bool enabled = true; };

// Grids are enabled on all available devices by default.
template< typename ConfigTag, typename Device > struct GridDeviceTag { static constexpr bool enabled = true; };
#ifndef HAVE_CUDA
template< typename ConfigTag > struct GridDeviceTag< ConfigTag, Devices::Cuda > { static constexpr bool enabled = false; };
#endif

// Grids are enabled only for the `int` and `long int` index types by default.
template< typename ConfigTag, typename Index > struct GridIndexTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct GridIndexTag< ConfigTag, int > { static constexpr bool enabled = true; };
template< typename ConfigTag > struct GridIndexTag< ConfigTag, long int > { static constexpr bool enabled = true; };

// The Grid is enabled for allowed dimensions and Real, Device and Index types.
//
// By specializing this tag you can enable or disable custom combinations of
// the grid template parameters. The default configuration is identical to the
// individual per-type tags.
template< typename ConfigTag, typename MeshType > struct GridTag { static constexpr bool enabled = false; };

template< typename ConfigTag, int Dimension, typename Real, typename Device, typename Index >
struct GridTag< ConfigTag, Grid< Dimension, Real, Device, Index > >
{
   static constexpr bool enabled = GridDimensionTag< ConfigTag, Dimension >::enabled  &&
                    GridRealTag< ConfigTag, Real >::enabled &&
                    GridDeviceTag< ConfigTag, Device >::enabled &&
                    GridIndexTag< ConfigTag, Index >::enabled;
};


// Configuration for unstructured meshes

// Meshes are enabled on all available devices by default.
template< typename ConfigTag, typename Device > struct MeshDeviceTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct MeshDeviceTag< ConfigTag, Devices::Host > { static constexpr bool enabled = true; };
#ifdef HAVE_CUDA
template< typename ConfigTag > struct MeshDeviceTag< ConfigTag, Devices::Cuda > { static constexpr bool enabled = true; };
#endif

// All available cell topologies are disabled by default.
template< typename ConfigTag, typename CellTopology > struct MeshCellTopologyTag { static constexpr bool enabled = false; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Wedge > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Pyramid > { static constexpr bool enabled = true; };
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };
// TODO: Simplex has not been tested yet
//template< typename ConfigTag > struct MeshCellTopologyTag< ConfigTag, Topologies::Simplex > { static constexpr bool enabled = true; };

// All sensible space dimensions are enabled by default.
template< typename ConfigTag, typename CellTopology, int SpaceDimension > struct MeshSpaceDimensionTag { static constexpr bool enabled = SpaceDimension >= CellTopology::dimension && SpaceDimension <= 3; };

// Meshes are enabled only for the `float` and `double` real types by default.
template< typename ConfigTag, typename Real > struct MeshRealTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct MeshRealTag< ConfigTag, float > { static constexpr bool enabled = true; };
template< typename ConfigTag > struct MeshRealTag< ConfigTag, double > { static constexpr bool enabled = true; };

// Meshes are enabled only for the `int` and `long int` global index types by default.
template< typename ConfigTag, typename GlobalIndex > struct MeshGlobalIndexTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct MeshGlobalIndexTag< ConfigTag, int > { static constexpr bool enabled = true; };
template< typename ConfigTag > struct MeshGlobalIndexTag< ConfigTag, long int > { static constexpr bool enabled = true; };

// Meshes are enabled only for the `short int` local index type by default.
template< typename ConfigTag, typename LocalIndex > struct MeshLocalIndexTag { static constexpr bool enabled = false; };
template< typename ConfigTag > struct MeshLocalIndexTag< ConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig to use.
template< typename ConfigTag >
struct MeshConfigTemplateTag
{
   template< typename Cell, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
   using MeshConfig = DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >;
};

// The Mesh is enabled for allowed Device, CellTopology, SpaceDimension, Real,
// GlobalIndex, LocalIndex and Id types as specified above.
//
// By specializing this tag you can enable or disable custom combinations of
// the grid template parameters. The default configuration is identical to the
// individual per-type tags.
//
// NOTE: We can't specialize the whole MeshType as it was done for the GridTag,
//       because we don't know the MeshConfig and the compiler can't deduce it
//       at the time of template specializations, so something like this does
//       not work:
//
//          struct MeshTag< ConfigTag,
//                      Mesh< typename MeshConfigTemplateTag< ConfigTag >::
//                         template MeshConfig< CellTopology, SpaceDimension, Real, GlobalIndex, LocalIndex > > >
//
template< typename ConfigTag, typename Device, typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
struct MeshTag
{
   static constexpr bool enabled =
            MeshDeviceTag< ConfigTag, Device >::enabled &&
            MeshCellTopologyTag< ConfigTag, CellTopology >::enabled &&
            MeshSpaceDimensionTag< ConfigTag, CellTopology, SpaceDimension >::enabled &&
            MeshRealTag< ConfigTag, Real >::enabled &&
            MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled &&
            MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled;
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace noa::TNL
