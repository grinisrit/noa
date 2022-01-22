// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Operators {   

/***
 * Default implementation for case when one differentiate with respect
 * to some other variable than x. In this case the result is zero.
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          int XDifference,
          int YDifference,
          int ZDifference,
          int XDirection,
          int YDirection,
          int ZDirection >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   XDifference, YDifference, ZDifference,
   XDirection, YDirection, ZDirection >
{
   static_assert( YDifference != 0 || ZDifference != 0,
      "You try to use default finite difference with 'wrong' template parameters. It means that required finite difference was not implmented yet." );
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         return 0.0;
      }
};

/****
 * 1st order forward difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighborEntities.template getEntityIndex< 1 >()] - u_c ) * hxDiv;
      }
};

/****
 * 1st order backward difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   -1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u_c - u[ neighborEntities.template getEntityIndex< -1 >()] ) * hxDiv;
      }
};

/****
 * 1st order central difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1 >();
         return ( u[ neighborEntities.template getEntityIndex< 1 >() ] -
                  u[ neighborEntities.template getEntityIndex< -1 >() ] ) * ( 0.5 * hxDiv );
      }
};

/****
 * 2nd order central difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighborEntities.template getEntityIndex< 2 >() ] -
                  2.0 * u_c +
                  u[ neighborEntities.template getEntityIndex< 1 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   -1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighborEntities.template getEntityIndex< -2 >() ] -
                  2.0 * u_c +
                  u[ neighborEntities.template getEntityIndex< -1 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighborEntities.template getEntityIndex< 1 >() ] -
                  2.0 * u_c +
                  u[ neighborEntities.template getEntityIndex< -1 >() ] ) * hxSquareDiv;
      }
};

} // namespace Operators
} // namespace noaTNL

