// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Operators/fdm/ForwardFiniteDifference.h>
#include <noa/3rdparty/TNL/Operators/fdm/BackwardFiniteDifference.h>
#include <noa/3rdparty/TNL/Operators/geometric/ExactGradientNorm.h>
#include <noa/3rdparty/TNL/Operators/Operator.h>

namespace noaTNL {
namespace Operators {   

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class TwoSidedGradientNorm
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TwoSidedGradientNorm< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
   : public Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 1, 1, Real, Index >
{
   public:
 
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactGradientNorm< 1, RealType > ExactOperatorType;
 
   TwoSidedGradientNorm()
   : epsSquare( 0.0 ){}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      ForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      return ::sqrt( this->epsSquare + 0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b ) );
   }
 
   void setEps( const Real& eps )
   {
      this->epsSquare = eps*eps;
   }
 
   private:
 
   RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TwoSidedGradientNorm< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
   : public Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 2, 2, Real, Index >
{
   public:
 
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactGradientNorm< 2, RealType > ExactOperatorType;
 
   TwoSidedGradientNorm()
   : epsSquare( 0.0 ){}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      ForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      ForwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YForwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      const RealType u_y_f = YForwardDifference( u, entity );
      const RealType u_y_b = YBackwardDifference( u, entity );
 
      return ::sqrt( this->epsSquare +
         0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                 u_y_f * u_y_f + u_y_b * u_y_b ) );
   }
 
   void setEps( const Real& eps )
   {
      this->epsSquare = eps*eps;
   }
 
 
   private:
 
   RealType epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class TwoSidedGradientNorm< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >,
                         Functions::MeshInteriorDomain, 3, 3, Real, Index >
{
   public:
 
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactGradientNorm< 3, RealType > ExactOperatorType;
 
   TwoSidedGradientNorm()
   : epsSquare( 0.0 ){}

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const
   {
      ForwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XForwardDifference;
      ForwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YForwardDifference;
      ForwardFiniteDifference< typename MeshEntity::MeshType, 0, 0, 1, Real, Index > ZForwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 1, 0, 0, Real, Index > XBackwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 0, 1, 0, Real, Index > YBackwardDifference;
      BackwardFiniteDifference< typename MeshEntity::MeshType, 0, 0, 1, Real, Index > ZBackwardDifference;
      const RealType u_x_f = XForwardDifference( u, entity );
      const RealType u_x_b = XBackwardDifference( u, entity );
      const RealType u_y_f = YForwardDifference( u, entity );
      const RealType u_y_b = YBackwardDifference( u, entity );
      const RealType u_z_f = ZForwardDifference( u, entity );
      const RealType u_z_b = ZBackwardDifference( u, entity );
 
      return ::sqrt( this->epsSquare +
         0.5 * ( u_x_f * u_x_f + u_x_b * u_x_b +
                 u_y_f * u_y_f + u_y_b * u_y_b +
                 u_z_f * u_z_f + u_z_b * u_z_b ) );
 
   }
 
 
   void setEps(const Real& eps)
   {
      this->epsSquare = eps*eps;
   }
 
   private:
 
   RealType epsSquare;
};

} // namespace Operators
} // namespace noaTNL

