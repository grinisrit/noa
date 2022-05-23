// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Operators {

/****
 * 1st order forward difference
 */
template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1, 0, 0, 1, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1, 0, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] - u_c ) * hxDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 1, 0, 0, 1, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] - u_c ) * hyDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 1, 0, 0, 1 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hzDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -1 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] - u_c ) * hzDiv;
   }
};

/****
 * 1st order backward difference
 */
template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1, 0, 0, -1, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1, 0, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u_c - u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] ) * hxDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 1, 0, 0, -1, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u_c - u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] ) * hyDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 1, 0, 0, -1 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hzDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -1 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u_c - u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] ) * hzDiv;
   }
};

/****
 * 1st order central difference
 */
template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 1, 0, 0, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1, 0, 0 >();
      return ( u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ]
               - u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] )
           * ( 0.5 * hxDiv );
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 1, 0, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1, 0 >();
      return ( u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ]
               - u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] )
           * ( 0.5 * hyDiv );
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 1, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hzDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -1 >();
      return ( u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ]
               - u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] )
           * ( 0.5 * hzDiv );
   }
};

/****
 * 2nd order central difference
 */
template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 2, 0, 0, 1, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 2, 0, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 2, 0, 0, -1, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< -2, 0, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 2, 0, 0, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 2, 0, 0, 1, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 2, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 2, 0, 0, -1, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, -2, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 2, 0, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hySquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] )
           * hySquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 2, 0, 0, 1 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 0, 2 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 2, 0, 0, -1 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 0, -2 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] )
           * hxSquareDiv;
   }
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class FiniteDifferences< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, 0, 0, 2, 0, 0, 0 >
{
public:
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   static Real
   getValue( const MeshFunction& u, const MeshEntity& entity )
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const Real& hzSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
      const Real& u_c = u[ entity.getIndex() ];
      return ( u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] - 2.0 * u_c
               + u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] )
           * hzSquareDiv;
   }
};

}  // namespace Operators
}  // namespace noa::TNL
