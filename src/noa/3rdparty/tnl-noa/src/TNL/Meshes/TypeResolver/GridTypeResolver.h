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
class GridTypeResolver
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
      resolveGridDimension( Reader& reader, Functor&& functor );

      // NOTE: We could disable the grids only by the GridTag, but doing the
      //       resolution for all subtypes is more flexible and also pretty
      //       good optimization of compilation times.

      template< int MeshDimension >
      static bool
      resolveReal( Reader& reader, Functor&& functor );

      template< int MeshDimension, typename Real >
      static bool
      resolveIndex( Reader& reader, Functor&& functor );

      template< int MeshDimension, typename Real, typename Index >
      static bool
      resolveGridType( Reader& reader, Functor&& functor );

      template< typename GridType >
      static bool
      resolveTerminate( Reader& reader, Functor&& functor );
   };
};

}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/GridTypeResolver.hpp>
