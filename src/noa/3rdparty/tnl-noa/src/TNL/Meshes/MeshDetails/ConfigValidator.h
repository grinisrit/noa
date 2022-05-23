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

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polyhedron.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename EntityTopology, typename DimensionTag >
class ConfigValidatorSubtopologyLayer
: public ConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   static_assert( ! MeshConfig::subentityStorage( EntityTopology::dimension, DimensionTag::value )
                     || MeshConfig::subentityStorage( EntityTopology::dimension, 0 ),
                  "entities of which subentities are stored must store their subvertices" );
   static_assert( ! MeshConfig::subentityStorage( EntityTopology::dimension, DimensionTag::value )
                     || MeshConfig::subentityStorage( EntityTopology::dimension, 0 ),
                  "entities that are stored as subentities must store their subvertices" );
};

template< typename MeshConfig, typename EntityTopology >
class ConfigValidatorSubtopologyLayer< MeshConfig, EntityTopology, DimensionTag< 0 > >
{};

template< typename MeshConfig, typename EntityTopology, typename DimensionTag >
class ConfigValidatorSupertopologyLayer
: public ConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, typename DimensionTag::Decrement >
{
   static_assert( ! MeshConfig::superentityStorage( EntityTopology::dimension, DimensionTag::value )
                     || MeshConfig::subentityStorage( EntityTopology::dimension, 0 ),
                  "entities of which superentities are stored must store their subvertices" );
   static_assert( ! MeshConfig::superentityStorage( EntityTopology::dimension, DimensionTag::value )
                     || MeshConfig::subentityStorage( EntityTopology::dimension, 0 ),
                  "entities that are stored as superentities must store their subvertices" );
};

template< typename MeshConfig, typename EntityTopology >
class ConfigValidatorSupertopologyLayer< MeshConfig, EntityTopology, DimensionTag< EntityTopology::dimension > >
{};

template< typename MeshConfig, int dimension >
class ConfigValidatorLayer : public ConfigValidatorLayer< MeshConfig, dimension - 1 >,
                             public ConfigValidatorSubtopologyLayer<
                                MeshConfig,
                                typename Topologies::Subtopology< typename MeshConfig::CellTopology, dimension >::Topology,
                                DimensionTag< dimension - 1 > >,
                             public ConfigValidatorSupertopologyLayer<
                                MeshConfig,
                                typename Topologies::Subtopology< typename MeshConfig::CellTopology, dimension >::Topology,
                                DimensionTag< MeshConfig::CellTopology::dimension > >
{};

template< typename MeshConfig >
class ConfigValidatorLayer< MeshConfig, 0 >
{};

template< typename MeshConfig >
class ConfigValidatorLayerCell
: public ConfigValidatorLayer< MeshConfig, MeshConfig::CellTopology::dimension - 1 >,
  public ConfigValidatorSubtopologyLayer< MeshConfig,
                                          typename MeshConfig::CellTopology,
                                          DimensionTag< MeshConfig::CellTopology::dimension - 1 > >
{
   using CellTopology = typename MeshConfig::CellTopology;

   static_assert( MeshConfig::subentityStorage( CellTopology::dimension, 0 ), "subvertices of cells must be stored" );

   static_assert( ! std::is_same< CellTopology, Topologies::Polyhedron >::value
                     || MeshConfig::subentityStorage( CellTopology::dimension, 2 ),
                  "faces of cells must be stored for polyhedral meshes" );
};

template< typename MeshConfig >
class ConfigValidator : public ConfigValidatorLayerCell< MeshConfig >
{
   static constexpr int meshDimension = MeshConfig::CellTopology::dimension;

   static_assert( 1 <= meshDimension, "zero dimensional meshes are not supported" );
   static_assert( meshDimension <= MeshConfig::spaceDimension, "space dimension must not be less than mesh dimension" );
};

}  // namespace Meshes
}  // namespace noa::TNL
