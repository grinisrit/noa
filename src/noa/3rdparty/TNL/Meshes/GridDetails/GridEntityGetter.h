// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
namespace Meshes {

template< typename Grid,
          typename GridEntity,
          int EntityDimension = GridEntity::getEntityDimension() >
class GridEntityGetter
{
   //static_assert( false, "Wrong mesh type or entity topology." );
};

/***
 * The main code is in template specializations in GridEntityIndexer.h
 */

} // namespace Meshes
} // namespace TNL

