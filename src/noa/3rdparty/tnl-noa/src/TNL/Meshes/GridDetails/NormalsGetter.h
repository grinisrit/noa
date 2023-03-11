// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/Permutations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/Functions.h>

namespace noa::TNL {
namespace Meshes {

template< typename Index, Index EntityDimension, Index GridDimension >
struct NormalsGetter
{
public:
   using Coordinate = TNL::Containers::StaticVector< GridDimension, Index >;
   using OrientationNormalsContainer =
      TNL::Containers::StaticVector< Templates::combination( EntityDimension, GridDimension ), Coordinate >;
   using Permutations =
      Templates::make_int_permutations< GridDimension,
                                        Templates::build_ones_pack< GridDimension - EntityDimension, GridDimension > >;

   template<
      int Orientation,
      std::enable_if_t<
         Templates::isInLeftClosedRightOpenInterval( 0, Orientation, Templates::combination( EntityDimension, GridDimension ) ),
         bool > = true >
   constexpr static Coordinate
   getNormals()
   {
      using Value = Templates::get< Orientation, Permutations >;

      return BuildNormals< Value >::build();
   }

private:
   template< class >
   struct BuildNormals;

   template< int... Values >
   struct BuildNormals< TNL::Meshes::Templates::int_pack< Values... > >
   {
   public:
      constexpr static Coordinate
      build()
      {
         return Coordinate( Values... );
      }
   };
};

}  // namespace Meshes
}  // namespace noa::TNL
