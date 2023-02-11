// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/LinearDiffusion.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Operators {

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename MeshEntity >
__cuda_callable__
inline Real
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::operator()( const PreimageFunction& u,
                                                                                            const MeshEntity& entity,
                                                                                            const Real& time ) const
{
   static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 1, "Wrong preimage function" );
   const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2 >();
   return ( u[ neighborEntities.template getEntityIndex< -1 >() ] - 2.0 * u[ entity.getIndex() ]
            + u[ neighborEntities.template getEntityIndex< 1 >() ] )
        * hxSquareInverse;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename MeshEntity >
__cuda_callable__
inline Index
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::getLinearSystemRowLength(
   const MeshType& mesh,
   const IndexType& index,
   const MeshEntity& entity ) const
{
   return 3;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
__cuda_callable__
inline void
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::setMatrixElements( const PreimageFunction& u,
                                                                                                   const MeshEntity& entity,
                                                                                                   const RealType& time,
                                                                                                   const RealType& tau,
                                                                                                   Matrix& matrix,
                                                                                                   Vector& b ) const
{
   static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 1, "Wrong preimage function" );
   const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
   const IndexType& index = entity.getIndex();
   auto matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2 >();
   matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1 >(), -lambdaX );
   matrixRow.setElement( 1, index, 2.0 * lambdaX );
   matrixRow.setElement( 2, neighborEntities.template getEntityIndex< 1 >(), -lambdaX );
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename EntityType >
__cuda_callable__
inline Index
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::getLinearSystemRowLength(
   const MeshType& mesh,
   const IndexType& index,
   const EntityType& entity ) const
{
   return 5;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename EntityType >
__cuda_callable__
inline Real
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::operator()( const PreimageFunction& u,
                                                                                            const EntityType& entity,
                                                                                            const Real& time ) const
{
   static_assert( EntityType::getEntityDimension() == 2, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 2, "Wrong preimage function" );
   const typename EntityType::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
   return ( u[ neighborEntities.template getEntityIndex< -1, 0 >() ] + u[ neighborEntities.template getEntityIndex< 1, 0 >() ] )
           * hxSquareInverse
        + ( u[ neighborEntities.template getEntityIndex< 0, -1 >() ] + u[ neighborEntities.template getEntityIndex< 0, 1 >() ] )
             * hySquareInverse
        - 2.0 * u[ entity.getIndex() ] * ( hxSquareInverse + hySquareInverse );
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
__cuda_callable__
inline void
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::setMatrixElements( const PreimageFunction& u,
                                                                                                   const MeshEntity& entity,
                                                                                                   const RealType& time,
                                                                                                   const RealType& tau,
                                                                                                   Matrix& matrix,
                                                                                                   Vector& b ) const
{
   static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 2, "Wrong preimage function" );
   const IndexType& index = entity.getIndex();
   auto matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
   const RealType lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
   const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
   matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, -1 >(), -lambdaY );
   matrixRow.setElement( 1, neighborEntities.template getEntityIndex< -1, 0 >(), -lambdaX );
   matrixRow.setElement( 2, index, 2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, neighborEntities.template getEntityIndex< 1, 0 >(), -lambdaX );
   matrixRow.setElement( 4, neighborEntities.template getEntityIndex< 0, 1 >(), -lambdaY );
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename EntityType >
__cuda_callable__
inline Real
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::operator()( const PreimageFunction& u,
                                                                                            const EntityType& entity,
                                                                                            const Real& time ) const
{
   static_assert( EntityType::getEntityDimension() == 3, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 3, "Wrong preimage function" );
   const typename EntityType::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
   const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
   return ( u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ]
            + u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] )
           * hxSquareInverse
        + ( u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ]
            + u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] )
             * hySquareInverse
        + ( u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ]
            + u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] )
             * hzSquareInverse
        - 2.0 * u[ entity.getIndex() ] * ( hxSquareInverse + hySquareInverse + hzSquareInverse );
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename EntityType >
__cuda_callable__
inline Index
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::getLinearSystemRowLength(
   const MeshType& mesh,
   const IndexType& index,
   const EntityType& entity ) const
{
   return 7;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
__cuda_callable__
inline void
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::setMatrixElements( const PreimageFunction& u,
                                                                                                   const MeshEntity& entity,
                                                                                                   const RealType& time,
                                                                                                   const RealType& tau,
                                                                                                   Matrix& matrix,
                                                                                                   Vector& b ) const
{
   static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimension() == 3, "Wrong preimage function" );
   const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
   const IndexType& index = entity.getIndex();
   auto matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
   const RealType lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
   const RealType lambdaZ = tau * entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
   matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, 0, -1 >(), -lambdaZ );
   matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, -1, 0 >(), -lambdaY );
   matrixRow.setElement( 2, neighborEntities.template getEntityIndex< -1, 0, 0 >(), -lambdaX );
   matrixRow.setElement( 3, index, 2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, neighborEntities.template getEntityIndex< 1, 0, 0 >(), -lambdaX );
   matrixRow.setElement( 5, neighborEntities.template getEntityIndex< 0, 1, 0 >(), -lambdaY );
   matrixRow.setElement( 6, neighborEntities.template getEntityIndex< 0, 0, 1 >(), -lambdaZ );
}

}  // namespace Operators
}  // namespace noa::TNL
