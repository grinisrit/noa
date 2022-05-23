// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Meshes {

template< typename Grid, int EntityDimension, typename EntityOrientation_, typename EntityProportions_ >
class GridEntityTopology
{
public:
   using GridType = Grid;

   static constexpr int meshDimension = GridType::getMeshDimension();

   static constexpr int entityDimension = EntityDimension;

   using EntityOrientation = EntityOrientation_;

   using EntityProportions = EntityProportions_;

   // TODO: restore when CUDA allows it
   // static_assert( meshDimension == EntityOrientation_::size,
   //                "Entity orientation is not a proper static multiindex." );
};

}  // namespace Meshes
}  // namespace noa::TNL
