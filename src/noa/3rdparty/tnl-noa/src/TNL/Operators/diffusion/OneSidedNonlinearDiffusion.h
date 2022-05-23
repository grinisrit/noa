// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
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

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/ExactNonlinearDiffusion.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename Nonlinearity,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class OneSidedNonlinearDiffusion
{};

template< typename MeshReal, typename Device, typename MeshIndex, typename Nonlinearity, typename Real, typename Index >
class OneSidedNonlinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
public:
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Nonlinearity NonlinearityType;
   typedef typename MeshType::template MeshEntity< MeshType::getMeshDimension() > CellType;
   typedef ExactNonlinearDiffusion< MeshType::getMeshDimension(), typename Nonlinearity::ExactOperatorType, Real >
      ExactOperatorType;

   OneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity ) : nonlinearity( nonlinearity ) {}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real
   operator()( const MeshFunction& u, const MeshEntity& entity, const RealType& time = 0.0 ) const
   {
      const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
      const typename MeshEntity::MeshType& mesh = entity.getMesh();
      const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2 >();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1 >();
      const RealType& u_c = u[ center ];
      const RealType u_x_f = ( u[ east ] - u_c );
      const RealType u_x_b = ( u_c - u[ west ] );
      return ( u_x_f * this->nonlinearity[ center ] - u_x_b * this->nonlinearity[ west ] ) * hx_div;
   }

   template< typename MeshEntity >
   __cuda_callable__
   Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const MeshEntity& entity ) const
   {
      return 3;
   }

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const
   {
      typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
      const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1 >();
      const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2 >();
      const RealType& nonlinearity_center = this->nonlinearity[ center ];
      const RealType& nonlinearity_west = this->nonlinearity[ west ];
      const RealType aCoef = -lambda_x * nonlinearity_west;
      const RealType bCoef = lambda_x * ( nonlinearity_center + nonlinearity_west );
      const RealType cCoef = -lambda_x * nonlinearity_center;
      matrixRow.setElement( 0, west, aCoef );
      matrixRow.setElement( 1, center, bCoef );
      matrixRow.setElement( 2, east, cCoef );
   }

public:
   const Nonlinearity& nonlinearity;
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Nonlinearity, typename Real, typename Index >
class OneSidedNonlinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
public:
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Nonlinearity NonlinearityType;
   typedef ExactNonlinearDiffusion< MeshType::getMeshDimension(), typename Nonlinearity::ExactOperatorType, Real >
      ExactOperatorType;

   OneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity ) : nonlinearity( nonlinearity ) {}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real
   operator()( const MeshFunction& u, const MeshEntity& entity, const RealType& time = 0.0 ) const
   {
      const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
      const typename MeshEntity::MeshType& mesh = entity.getMesh();
      const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
      const RealType& hy_div = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1, 0 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1, 0 >();
      const IndexType& north = neighborEntities.template getEntityIndex< 0, 1 >();
      const IndexType& south = neighborEntities.template getEntityIndex< 0, -1 >();
      const RealType& u_c = u[ center ];
      const RealType u_x_f = ( u[ east ] - u_c );
      const RealType u_x_b = ( u_c - u[ west ] );
      const RealType u_y_f = ( u[ north ] - u_c );
      const RealType u_y_b = ( u_c - u[ south ] );

      const RealType& nonlinearity_center = this->nonlinearity[ center ];
      return ( u_x_f * nonlinearity_center - u_x_b * this->nonlinearity[ west ] ) * hx_div
           + ( u_y_f * nonlinearity_center - u_y_b * this->nonlinearity[ south ] ) * hy_div;
   }

   template< typename MeshEntity >
   __cuda_callable__
   Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const MeshEntity& entity ) const
   {
      return 5;
   }

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const
   {
      typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
      const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1, 0 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1, 0 >();
      const IndexType& north = neighborEntities.template getEntityIndex< 0, 1 >();
      const IndexType& south = neighborEntities.template getEntityIndex< 0, -1 >();
      const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
      const RealType lambda_y = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
      const RealType& nonlinearity_center = this->nonlinearity[ center ];
      const RealType& nonlinearity_west = this->nonlinearity[ west ];
      const RealType& nonlinearity_south = this->nonlinearity[ south ];
      const RealType aCoef = -lambda_y * nonlinearity_south;
      const RealType bCoef = -lambda_x * nonlinearity_west;
      const RealType cCoef =
         lambda_x * ( nonlinearity_center + nonlinearity_west ) + lambda_y * ( nonlinearity_center + nonlinearity_south );
      const RealType dCoef = -lambda_x * nonlinearity_center;
      const RealType eCoef = -lambda_y * nonlinearity_center;
      matrixRow.setElement( 0, south, aCoef );
      matrixRow.setElement( 1, west, bCoef );
      matrixRow.setElement( 2, center, cCoef );
      matrixRow.setElement( 3, east, dCoef );
      matrixRow.setElement( 4, north, eCoef );
   }

public:
   const Nonlinearity& nonlinearity;
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Nonlinearity, typename Real, typename Index >
class OneSidedNonlinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Nonlinearity, Real, Index >
{
public:
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Nonlinearity NonlinearityType;
   typedef ExactNonlinearDiffusion< MeshType::getMeshDimension(), typename Nonlinearity::ExactOperatorType, Real >
      ExactOperatorType;

   OneSidedNonlinearDiffusion( const Nonlinearity& nonlinearity ) : nonlinearity( nonlinearity ) {}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real
   operator()( const MeshFunction& u, const MeshEntity& entity, const RealType& time = 0.0 ) const
   {
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const typename MeshEntity::MeshType& mesh = entity.getMesh();
      const RealType& hx_div = entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
      const RealType& hy_div = entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
      const RealType& hz_div = entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1, 0, 0 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1, 0, 0 >();
      const IndexType& north = neighborEntities.template getEntityIndex< 0, 1, 0 >();
      const IndexType& south = neighborEntities.template getEntityIndex< 0, -1, 0 >();
      const IndexType& up = neighborEntities.template getEntityIndex< 0, 0, 1 >();
      const IndexType& down = neighborEntities.template getEntityIndex< 0, 0, -1 >();

      const RealType& u_c = u[ center ];
      const RealType u_x_f = ( u[ east ] - u_c );
      const RealType u_x_b = ( u_c - u[ west ] );
      const RealType u_y_f = ( u[ north ] - u_c );
      const RealType u_y_b = ( u_c - u[ south ] );
      const RealType u_z_f = ( u[ up ] - u_c );
      const RealType u_z_b = ( u_c - u[ down ] );

      const RealType& nonlinearity_center = this->nonlinearity[ center ];
      return ( u_x_f * nonlinearity_center - u_x_b * this->nonlinearity[ west ] ) * hx_div
           + ( u_y_f * nonlinearity_center - u_y_b * this->nonlinearity[ south ] ) * hx_div
           + ( u_z_f * nonlinearity_center - u_z_b * this->nonlinearity[ down ] ) * hz_div;
   }

   template< typename MeshEntity >
   __cuda_callable__
   Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const MeshEntity& entity ) const
   {
      return 7;
   }

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const
   {
      typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
      const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
      const IndexType& center = entity.getIndex();
      const IndexType& east = neighborEntities.template getEntityIndex< 1, 0, 0 >();
      const IndexType& west = neighborEntities.template getEntityIndex< -1, 0, 0 >();
      const IndexType& north = neighborEntities.template getEntityIndex< 0, 1, 0 >();
      const IndexType& south = neighborEntities.template getEntityIndex< 0, -1, 0 >();
      const IndexType& up = neighborEntities.template getEntityIndex< 0, 0, 1 >();
      const IndexType& down = neighborEntities.template getEntityIndex< 0, 0, -1 >();

      const RealType lambda_x = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
      const RealType lambda_y = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
      const RealType lambda_z = tau * entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
      const RealType& nonlinearity_center = this->nonlinearity[ center ];
      const RealType& nonlinearity_west = this->nonlinearity[ west ];
      const RealType& nonlinearity_south = this->nonlinearity[ south ];
      const RealType& nonlinearity_down = this->nonlinearity[ down ];
      const RealType aCoef = -lambda_z * nonlinearity_down;
      const RealType bCoef = -lambda_y * nonlinearity_south;
      const RealType cCoef = -lambda_x * nonlinearity_west;
      const RealType dCoef = lambda_x * ( nonlinearity_center + nonlinearity_west )
                           + lambda_y * ( nonlinearity_center + nonlinearity_south )
                           + lambda_z * ( nonlinearity_center + nonlinearity_down );
      const RealType eCoef = -lambda_x * nonlinearity_center;
      const RealType fCoef = -lambda_y * nonlinearity_center;
      const RealType gCoef = -lambda_z * nonlinearity_center;
      matrixRow.setElement( 0, down, aCoef );
      matrixRow.setElement( 1, south, bCoef );
      matrixRow.setElement( 2, west, cCoef );
      matrixRow.setElement( 3, center, dCoef );
      matrixRow.setElement( 4, east, eCoef );
      matrixRow.setElement( 5, north, fCoef );
      matrixRow.setElement( 5, up, gCoef );
   }

public:
   const Nonlinearity& nonlinearity;
};

}  // namespace Operators
}  // namespace noa::TNL
