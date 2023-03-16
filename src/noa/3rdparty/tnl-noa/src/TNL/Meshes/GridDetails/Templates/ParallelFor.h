// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>

namespace noa::TNL {
namespace Meshes {
namespace Templates {

/**
 * A dimension-based interface of ParallelFor algorithm
 */
template< int, typename, typename >
struct ParallelFor;

template< typename Device, typename Index >
struct ParallelFor< 1, Device, Index >
{
public:
   template< typename Func, typename... FuncArgs >
   inline static void
   exec( const TNL::Containers::StaticVector< 1, Index >& from,
         const TNL::Containers::StaticVector< 1, Index >& to,
         Func func,
         FuncArgs... args )
   {
      auto groupIndex = [ = ] __cuda_callable__( Index i, FuncArgs... args ) mutable
      {
         func( TNL::Containers::StaticVector< 1, Index >( i ), args... );
      };

      TNL::Algorithms::ParallelFor< Device >::exec( from.x(), to.x(), groupIndex, args... );
   }
};

template< typename Device, typename Index >
struct ParallelFor< 2, Device, Index >
{
public:
   template< typename Func, typename... FuncArgs >
   inline static void
   exec( const TNL::Containers::StaticVector< 2, Index >& from,
         const TNL::Containers::StaticVector< 2, Index >& to,
         Func func,
         FuncArgs... args )
   {
      auto groupIndex = [ = ] __cuda_callable__( Index i, Index j, FuncArgs... args ) mutable
      {
         func( TNL::Containers::StaticVector< 2, Index >( i, j ), args... );
      };

      TNL::Algorithms::ParallelFor2D< Device >::exec( from.x(), from.y(), to.x(), to.y(), groupIndex, args... );
   }
};

template< typename Device, typename Index >
struct ParallelFor< 3, Device, Index >
{
public:
   template< typename Func, typename... FuncArgs >
   inline static void
   exec( const TNL::Containers::StaticVector< 3, Index >& from,
         const TNL::Containers::StaticVector< 3, Index >& to,
         Func func,
         FuncArgs... args )
   {
      auto groupIndex = [ = ] __cuda_callable__( Index i, Index j, Index k, FuncArgs... args ) mutable
      {
         func( TNL::Containers::StaticVector< 3, Index >( i, j, k ), args... );
      };

      TNL::Algorithms::ParallelFor3D< Device >::exec(
         from.x(), from.y(), from.z(), to.x(), to.y(), to.z(), groupIndex, args... );
   }
};

}  // namespace Templates
}  // namespace Meshes
}  // namespace noa::TNL
