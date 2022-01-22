// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Meshes {


template< typename Grid,
          int EntityDimension,
          typename EntityOrientation_,
          typename EntityProportions_ >
class GridEntityTopology
{
   public:
 
      typedef Grid GridType;
 
      static constexpr int meshDimension = GridType::getMeshDimension();
 
      static constexpr int entityDimension = EntityDimension;
 
      typedef EntityOrientation_ EntityOrientation;
 
      typedef EntityProportions_ EntityProportions;
 
      // TODO: restore when CUDA allows it
   //static_assert( meshDimension == EntityOrientation_::size,
   //               "Entity orientation is not a proper static multiindex." );
};

} // namespace Meshes
} // namespace noaTNL

