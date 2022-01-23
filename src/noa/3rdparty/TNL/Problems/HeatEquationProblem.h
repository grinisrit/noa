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

#include <noa/3rdparty/TNL/Problems/PDEProblem.h>
#include <noa/3rdparty/TNL/Operators/diffusion/LinearDiffusion.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionView.h>
#include <noa/3rdparty/TNL/Timer.h>
#include <noa/3rdparty/TNL/Solvers/PDE/ExplicitUpdater.h>
#include <noa/3rdparty/TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <noa/3rdparty/TNL/Solvers/PDE/BackwardTimeDiscretisation.h>

#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>

namespace noa::TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = Operators::LinearDiffusion< Mesh,
                                                              typename BoundaryCondition::RealType > >
class HeatEquationProblem : public PDEProblem< Mesh,
                                               typename Mesh::RealType,
                                               typename Mesh::DeviceType,
                                               typename Mesh::IndexType  >
{
   public:

      typedef typename Mesh::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename Mesh::IndexType IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;
      typedef Pointers::SharedPointer<  DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MatrixType;

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool writeEpilog( Logger& logger );


      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofs );

      template< typename MatrixPointer >
      bool setupLinearSystem( MatrixPointer& matrixPointer );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofs );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofs );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _u,
                              DofVectorPointer& _fu );

      void applyBoundaryConditions( const RealType& time,
                                    DofVectorPointer& dofs );

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 DofVectorPointer& dofsPointer,
                                 MatrixPointer& matrixPointer,
                                 DofVectorPointer& rightHandSidePointer );

   protected:

      using DistributedMeshSynchronizerType = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Meshes::DistributedMeshes::DistributedMesh< typename MeshFunctionType::MeshType > >;
      DistributedMeshSynchronizerType synchronizer;

      MeshFunctionPointer uPointer;
      MeshFunctionPointer fuPointer;

      DifferentialOperatorPointer differentialOperatorPointer;

      BoundaryConditionPointer boundaryConditionPointer;

      RightHandSidePointer rightHandSidePointer;

      Timer gpuTransferTimer;

      Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;

      Solvers::PDE::LinearSystemAssembler< Mesh,
                                           MeshFunctionType,
                                           DifferentialOperator,
                                           BoundaryCondition,
                                           RightHandSide,
                                           Solvers::PDE::BackwardTimeDiscretisation,
                                           DofVectorType > systemAssembler;

     bool catchExceptions = true;
};

} // namespace Problems
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Problems/HeatEquationProblem_impl.h>
