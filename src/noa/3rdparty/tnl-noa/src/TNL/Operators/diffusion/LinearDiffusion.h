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

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Operator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/ExactLinearDiffusion.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh, typename Real = typename Mesh::RealType, typename Index = typename Mesh::GlobalIndexType >
class LinearDiffusion
{};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
: public Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Functions::MeshInteriorDomain, 1, 1, Real, Index >
{
public:
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactLinearDiffusion< 1 > ExactOperatorType;

   static const int Dimension = MeshType::getMeshDimension();

   static constexpr int
   getMeshDimension()
   {
      return Dimension;
   }

   template< typename PreimageFunction, typename MeshEntity >
   __cuda_callable__
   inline Real
   operator()( const PreimageFunction& u, const MeshEntity& entity, const RealType& time = 0.0 ) const;

   template< typename MeshEntity >
   __cuda_callable__
   inline Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const MeshEntity& entity ) const;

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const;
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
: public Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Functions::MeshInteriorDomain, 2, 2, Real, Index >
{
public:
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactLinearDiffusion< 2 > ExactOperatorType;

   static const int Dimension = MeshType::getMeshDimension();

   static constexpr int
   getMeshDimension()
   {
      return Dimension;
   }

   template< typename PreimageFunction, typename EntityType >
   __cuda_callable__
   inline Real
   operator()( const PreimageFunction& u, const EntityType& entity, const Real& time = 0.0 ) const;

   template< typename EntityType >
   __cuda_callable__
   inline Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const EntityType& entity ) const;

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const;
};

template< typename MeshReal, typename Device, typename MeshIndex, typename Real, typename Index >
class LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
: public Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Functions::MeshInteriorDomain, 3, 3, Real, Index >
{
public:
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef ExactLinearDiffusion< 3 > ExactOperatorType;

   static const int Dimension = MeshType::getMeshDimension();

   static constexpr int
   getMeshDimension()
   {
      return Dimension;
   }

   template< typename PreimageFunction, typename EntityType >
   __cuda_callable__
   inline Real
   operator()( const PreimageFunction& u, const EntityType& entity, const Real& time = 0.0 ) const;

   template< typename EntityType >
   __cuda_callable__
   inline Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const EntityType& entity ) const;

   template< typename PreimageFunction, typename MeshEntity, typename Matrix, typename Vector >
   __cuda_callable__
   inline void
   setMatrixElements( const PreimageFunction& u,
                      const MeshEntity& entity,
                      const RealType& time,
                      const RealType& tau,
                      Matrix& matrix,
                      Vector& b ) const;
};

}  // namespace Operators
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/LinearDiffusion_impl.h>
