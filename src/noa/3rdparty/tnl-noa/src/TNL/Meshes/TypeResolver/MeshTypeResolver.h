// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag, typename Device >
class MeshTypeResolver
{
public:
   template< typename Reader, typename Functor >
   static bool
   run( Reader& reader, Functor&& functor );

protected:
   template< typename Reader, typename Functor >
   struct detail
   {
      static bool
      resolveCellTopology( Reader& reader, Functor&& functor );

      // NOTE: We could disable the meshes only by the MeshTag, but doing the
      //       resolution for all subtypes is more flexible and also pretty
      //       good optimization of compilation times.

      template< typename CellTopology >
      static bool
      resolveSpaceDimension( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension >
      static bool
      resolveReal( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real >
      static bool
      resolveGlobalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex >
      static bool
      resolveLocalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
      static bool
      resolveMeshType( Reader& reader, Functor&& functor );

      template< typename MeshConfig >
      static bool
      resolveTerminate( Reader& reader, Functor&& functor );
   };
};

}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/MeshTypeResolver.hpp>
