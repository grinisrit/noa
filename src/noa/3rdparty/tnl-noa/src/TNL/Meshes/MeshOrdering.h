// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>

namespace noa::TNL {
namespace Meshes {

// general implementation covering grids
template< typename Mesh, typename Ordering >
struct MeshOrdering
{
   void
   reorder( Mesh& mesh )
   {}
};

// reordering makes sense only for unstructured meshes
template< typename MeshConfig, typename Device, typename Ordering >
struct MeshOrdering< TNL::Meshes::Mesh< MeshConfig, Device >, Ordering >
{
   using Mesh = TNL::Meshes::Mesh< MeshConfig, Device >;

   void
   reorder( Mesh& mesh )
   {
      Algorithms::staticFor< int, 0, Mesh::getMeshDimension() >(
         [ & ]( auto i )
         {
            // make sure to reorder cells first
            constexpr int dim = Mesh::getMeshDimension() - i;
            using PermutationArray = typename Mesh::GlobalIndexArray;
            PermutationArray perm;
            PermutationArray iperm;
            using EntityType = typename Mesh::template EntityType< dim >;
            Ordering::template getPermutations< EntityType >( mesh, perm, iperm );
            mesh.template reorderEntities< dim >( perm, iperm );
         } );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
