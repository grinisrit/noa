// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Meshes {

enum GridEntityStencilStorage
{ 
   GridEntityNoStencil = 0,
   GridEntityCrossStencil,
   GridEntityFullStencil
};

template< int storage >
class GridEntityStencilStorageTag
{
   public:
 
      static constexpr int stencilStorage = storage;
};

/****
 * This class says what neighbor grid entity indexes shall be pre-computed and stored in the
 * grid entity structure. If neighborEntityStorage() returns false, nothing is stored.
 * Otherwise, if neighbor entity storage is enabled, we may store either only neighbor entities in a cross like this
 *
 *                X
 *   X            X
 *  XOX    or   XXOXX   etc.
 *   X            X
 *                X
 *
 * or all neighbor entities like this
 *
 *           XXXXX
 *  XXX      XXXXX
 *  XOX  or  XXOXX  etc.
 *  XXX      XXXXX
 *           XXXXX
 */

class GridEntityNoStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighborEntityStorage( int neighborEntityStorage )
      {
         return false;
      }
 
      constexpr static int getStencilSize()
      {
         return 0;
      }
};

template< int stencilSize = 1 >
class GridEntityCrossStencilStorage
{
   public:
 
      template< typename GridEntity >
      constexpr static bool neighborEntityStorage( const int neighborEntityDimension )
      {
         return ( GridEntity::getEntityDimension() == GridEntity::GridType::getMeshDimension() &&
                  neighborEntityDimension == GridEntity::GridType::getMeshDimension() );
      }
 
      constexpr static int getStencilSize()
      {
         return stencilSize;
      }
};

} // namespace Meshes
} // namespace noaTNL
