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

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename NonlinearDiffusionOperator,
	  typename OperatorQ,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class FiniteVolumeNonlinearOperator
{
 
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class FiniteVolumeNonlinearOperator< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;

   template< typename MeshEntity,
             typename Vector >
   __cuda_callable__
   Real operator()( const MeshEntity& entity,
                    const Vector& u,
                    const RealType& time) const;
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename MeshEntity,
             typename MeshFunction,
             typename Vector,
             typename Matrix >
   __cuda_callable__
      void setMatrixElements( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;
   
   public:
   
   OperatorQ operatorQ;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class FiniteVolumeNonlinearOperator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;
   

   template< typename MeshEntity,
             typename Vector >
   __cuda_callable__
   Real operator()( const MeshEntity& entity,
                    const Vector& u,
                    const RealType& time) const;
   
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename MeshEntity,
             typename MeshFunction,
             typename Vector,
             typename Matrix >
   __cuda_callable__
      void setMatrixElements( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;
   
   public:
   
   OperatorQ operatorQ;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename OperatorQ >
class FiniteVolumeNonlinearOperator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, OperatorQ, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef OperatorQ OperatorQType;

   template< typename MeshEntity, 
             typename Vector >
   __cuda_callable__
   Real operator()( const MeshEntity& entity,
                    const Vector& u,
                    const RealType& time) const;
   
   template< typename MeshEntity >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const MeshEntity& entity ) const;

   template< typename MeshEntity,
             typename MeshFunction,
             typename Vector,
             typename Matrix >
   __cuda_callable__
      void setMatrixElements( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunction& u,
                               Vector& b,
                               Matrix& matrix ) const;
   
   public:
   
   OperatorQ operatorQ;
};

} // namespace Operators
} // namespace noa::TNL

#include "FiniteVolumeNonlinearOperator_impl.h"
