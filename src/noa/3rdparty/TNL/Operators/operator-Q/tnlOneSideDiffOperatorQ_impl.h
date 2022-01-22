// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noaTNL {
namespace Operators {

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps*eps;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,          
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_x = ( u[ neighborEntities.template getEntityIndex< 1 >() ] - u[ cellIndex ] ) *
                         mesh.template getSpaceStepsProducts< -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x );          
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType& u_x_f = ( u[ neighborEntities.template getEntityIndex< 1 >() ] - u_c ) * 
                           mesh.template getSpaceStepsProducts< -1 >();
   const RealType& u_x_b = ( u_c - u[ neighborEntities.template getEntityIndex< -1 >() ] ) * 
                           mesh.template getSpaceStepsProducts< -1 >();   
   return ::sqrt( this->epsSquare + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b ) );
}

/***
 * 2D
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps*eps;
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,          
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType u_x = ( u[ neighborEntities.template getEntityIndex< 1, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y = ( u[ neighborEntities.template getEntityIndex< 0, 1 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y ); 
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   const RealType u_x_f = ( u[ neighborEntities.template getEntityIndex< 1, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y_f = ( u[ neighborEntities.template getEntityIndex< 0, 1 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, -1 >();
   const RealType u_x_b = ( u_c - u[ neighborEntities.template getEntityIndex< -1, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< -1, 0 >();
   const RealType u_y_b = ( u_c - u[ neighborEntities.template getEntityIndex< 0, -1 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, -1 >();
   
   return ::sqrt( this->epsSquare + 
                0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                        u_y_f * u_y_f + u_y_b * u_y_b ) );
}
/***
 * 3D
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
void 
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setEps( const Real& eps )
{
  this->eps = eps;
  this->epsSquare = eps * eps;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,            
            const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c =u[ cellIndex ];
   
   const RealType u_x = ( u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y = ( u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z = ( u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] - u_c ) *
                         mesh.template getSpaceStepsProducts< 0, 0, -1 >();
   return ::sqrt( this->epsSquare + u_x * u_x + u_y * u_y + u_z * u_z ); 
}
   
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getValueStriped( const MeshFunction& u,
                 const MeshEntity& entity,                 
                 const Real& time ) const
{
   const IndexType& cellIndex = entity.getIndex();
   const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();      
   const typename MeshEntity::MeshType& mesh = entity.getMesh();
   const RealType& u_c = u[ cellIndex ];
   
   const RealType u_x_f = ( u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y_f = ( u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z_f = ( u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] - u_c ) *
                          mesh.template getSpaceStepsProducts< 0, 0, -1 >();   
   const RealType u_x_b = ( u_c - u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< -1, 0, 0 >();
   const RealType u_y_b = ( u_c - u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, -1, 0 >();
   const RealType u_z_b = ( u_c - u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] ) *
                          mesh.template getSpaceStepsProducts< 0, 0, -1 >();
   
   return ::sqrt( this->epsSquare + 
                0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                        u_y_f * u_y_f + u_y_b * u_y_b + 
                        u_z_f * u_z_f + u_z_b * u_z_b ) );
}   

} // namespace Operators
} // namespace noaTNL
