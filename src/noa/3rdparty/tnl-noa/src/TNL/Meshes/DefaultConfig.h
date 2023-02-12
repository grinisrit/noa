// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexMap.h>

namespace noa::TNL {
namespace Meshes {

/****
 * Basic structure for mesh configuration.
 */
template< typename Cell,
          int SpaceDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = short int >
struct DefaultConfig
{
   using CellTopology = Cell;
   using RealType = Real;
   using GlobalIndexType = GlobalIndex;
   using LocalIndexType = LocalIndex;

   static constexpr int spaceDimension = SpaceDimension;
   static constexpr int meshDimension = Cell::dimension;

   /****
    * Storage of subentities of mesh entities.
    */
   static constexpr bool
   subentityStorage( int entityDimension, int subentityDimension )
   {
      return true;
      // Subvertices must be stored for all entities which appear in other
      // subentity or superentity mappings.
      // return SubentityDimension == 0;
   }

   /****
    * Storage of superentities of mesh entities.
    */
   static constexpr bool
   superentityStorage( int entityDimension, int superentityDimension )
   {
      return true;
   }

   /****
    * Storage of mesh entity tags. Boundary tags are necessary for the mesh traverser.
    *
    * The configuration must satisfy the following necessary conditions in
    * order to provide boundary tags:
    *    - faces must store the cell indices in the superentity layer
    *    - if dim(entity) < dim(face), the entities on which the tags are stored
    *      must be stored as subentities of faces
    */
   static constexpr bool
   entityTagsStorage( int entityDimension )
   {
      return superentityStorage( meshDimension - 1, meshDimension )
          && ( entityDimension >= meshDimension - 1 || subentityStorage( meshDimension - 1, entityDimension ) );
      // return false;
   }

   /****
    * Storage of the dual graph.
    *
    * If enabled, links from vertices to cells must be stored.
    */
   static constexpr bool
   dualGraphStorage()
   {
      return true;
   }

   /****
    * Cells must have at least this number of common vertices to be considered
    * as neighbors in the dual graph.
    */
   static constexpr int dualGraphMinCommonVertices = meshDimension;
};

}  // namespace Meshes
}  // namespace noa::TNL
