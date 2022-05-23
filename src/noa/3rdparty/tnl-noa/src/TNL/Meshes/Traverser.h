// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>

namespace noa::TNL {
namespace Meshes {

template< typename Mesh,
          typename MeshEntity,
          // extra parameter which is used only for specializations implementing grid traversers
          int EntitiesDimension = MeshEntity::getEntityDimension() >
class Traverser
{
public:
   using MeshType = Mesh;
   using MeshPointer = Pointers::SharedPointer< MeshType >;
   using DeviceType = typename MeshType::DeviceType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;

   template< typename EntitiesProcessor, typename UserData >
   void
   processBoundaryEntities( const MeshPointer& meshPointer, UserData userData ) const;

   template< typename EntitiesProcessor, typename UserData >
   void
   processInteriorEntities( const MeshPointer& meshPointer, UserData userData ) const;

   template< typename EntitiesProcessor, typename UserData >
   void
   processAllEntities( const MeshPointer& meshPointer, UserData userData ) const;

   template< typename EntitiesProcessor, typename UserData >
   void
   processGhostEntities( const MeshPointer& meshPointer, UserData userData ) const;

   template< typename EntitiesProcessor, typename UserData >
   void
   processLocalEntities( const MeshPointer& meshPointer, UserData userData ) const;
};

}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Traverser.hpp>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid3D.h>
