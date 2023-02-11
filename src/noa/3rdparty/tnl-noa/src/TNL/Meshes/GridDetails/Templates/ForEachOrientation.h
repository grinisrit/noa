// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NormalsGetter.h>

namespace noa::TNL {
namespace Meshes {
namespace Templates {

template< typename Index, int Orientation, int EntityDimension, int Dimension, int SkipValue >
struct _ForEachOrientationMain;

template< typename Index, int Orientation, int EntityDimension, int Dimension, int SkipValue >
struct _ForEachOrientationSupport
{
   using NormalsGetterType = NormalsGetter< Index, EntityDimension, Dimension >;

public:
   template< typename Func >
   inline static void
   exec( Func func )
   {
      func( std::integral_constant< int, Orientation >(), NormalsGetterType::template getNormals< Orientation >() );

      _ForEachOrientationMain< Index, Orientation - 1, EntityDimension, Dimension, SkipValue >::exec( func );
   }
};

template< typename Index, int EntityDimension, int Dimension, int SkipValue >
struct _ForEachOrientationSupport< Index, 0, EntityDimension, Dimension, SkipValue >
{
public:
   using NormalsGetterType = NormalsGetter< Index, EntityDimension, Dimension >;

   template< typename Func >
   inline static void
   exec( Func func )
   {
      func( std::integral_constant< int, 0 >(), NormalsGetterType::template getNormals< 0 >() );
   }
};

template< typename Index, int EntityDimension, int Dimension >
struct _ForEachOrientationSupport< Index, 0, EntityDimension, Dimension, 0 >
{
public:
   template< typename Func >
   inline static void
   exec( Func func )
   {}
};

template< typename Index, int Orientation, int EntityDimension, int Dimension, int SkipValue >
struct _ForEachOrientationMain
: std::conditional_t<
     Orientation == SkipValue,
     _ForEachOrientationSupport< Index, ( Orientation <= 1 ? 0 : Orientation - 1 ), EntityDimension, Dimension, SkipValue >,
     _ForEachOrientationSupport< Index, Orientation, EntityDimension, Dimension, SkipValue > >
{};

template< typename Index, int EntityDimension, int Dimension, int skipOrientation = -1 >
struct ForEachOrientation
: _ForEachOrientationMain< Index, combination( EntityDimension, Dimension ) - 1, EntityDimension, Dimension, skipOrientation >
{};
}  // namespace Templates
}  // namespace Meshes
}  // namespace noa::TNL
