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

#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Operator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/FunctionInverseOperator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/geometric/FDMGradientNorm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/NeumannBoundaryConditions.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/OneSidedNonlinearDiffusion.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/OperatorFunction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/diffusion/ExactMeanCurvature.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType,
          bool EvaluateNonlinearityOnFly = false >
class OneSidedMeanCurvature
: public Operator< Mesh, Functions::MeshInteriorDomain, Mesh::getMeshDimension(), Mesh::getMeshDimension(), Real, Index >
{
public:
   typedef Mesh MeshType;
   typedef Pointers::SharedPointer< MeshType > MeshPointer;
   typedef Real RealType;
   typedef Index IndexType;
   typedef FDMGradientNorm< MeshType, ForwardFiniteDifference, RealType, IndexType > GradientNorm;
   typedef FunctionInverseOperator< GradientNorm > NonlinearityOperator;
   typedef Functions::MeshFunction< MeshType, MeshType::getMeshDimension(), RealType > NonlinearityMeshFunction;
   typedef Functions::Analytic::Constant< MeshType::getMeshDimension(), RealType > NonlinearityBoundaryConditionsFunction;
   typedef NeumannBoundaryConditions< MeshType, NonlinearityBoundaryConditionsFunction > NonlinearityBoundaryConditions;
   typedef Functions::OperatorFunction< NonlinearityOperator,
                                        NonlinearityMeshFunction,
                                        NonlinearityBoundaryConditions,
                                        EvaluateNonlinearityOnFly >
      Nonlinearity;
   typedef OneSidedNonlinearDiffusion< Mesh, Nonlinearity, RealType, IndexType > NonlinearDiffusion;
   typedef ExactMeanCurvature< Mesh::getMeshDimension(), RealType > ExactOperatorType;

   OneSidedMeanCurvature( const MeshPointer& meshPointer )
   : nonlinearityOperator( gradientNorm ), nonlinearity( nonlinearityOperator, nonlinearityBoundaryConditions, meshPointer ),
     nonlinearDiffusion( nonlinearity )
   {}

   void
   setRegularizationEpsilon( const RealType& eps )
   {
      this->gradientNorm.setEps( eps );
   }

   void
   setPreimageFunction( typename Nonlinearity::PreimageFunctionType& preimageFunction )
   {
      this->nonlinearity.setPreimageFunction( preimageFunction );
   }

   bool
   refresh( const RealType& time = 0.0 )
   {
      return this->nonlinearity.refresh( time );
   }

   bool
   deepRefresh( const RealType& time = 0.0 )
   {
      return this->nonlinearity.deepRefresh( time );
   }

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real
   operator()( const MeshFunction& u, const MeshEntity& entity, const RealType& time = 0.0 ) const
   {
      return this->nonlinearDiffusion( u, entity, time );
   }

   template< typename MeshEntity >
   __cuda_callable__
   Index
   getLinearSystemRowLength( const MeshType& mesh, const IndexType& index, const MeshEntity& entity ) const
   {
      return this->nonlinearDiffusion.getLinearSystemRowLength( mesh, index, entity );
   }

   template< typename MeshEntity, typename MeshFunction, typename Vector, typename Matrix >
   __cuda_callable__
   void
   setMatrixElements( const RealType& time,
                      const RealType& tau,
                      const MeshType& mesh,
                      const IndexType& index,
                      const MeshEntity& entity,
                      const MeshFunction& u,
                      Vector& b,
                      Matrix& matrix ) const
   {
      this->nonlinearDiffusion.setMatrixElements( time, tau, mesh, index, entity, u, b, matrix );
   }

protected:
   NonlinearityBoundaryConditions nonlinearityBoundaryConditions;

   GradientNorm gradientNorm;

   NonlinearityOperator nonlinearityOperator;

   Nonlinearity nonlinearity;

   NonlinearDiffusion nonlinearDiffusion;
};

}  // namespace Operators
}  // namespace noa::TNL
