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

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Logger.h>
#include <TNL/Solvers/PDE/BoundaryConditionsSetter.h>
#include <TNL/MPI/Wrappers.h>

#include "HeatEquationProblem.h"

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Heat equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeEpilog( Logger& logger )
{
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, "boundary-conditions-" ) )
   {
      std::cerr << "I was not able to initialize the boundary conditions." << std::endl;
      return false;
   }
   if( ! this->rightHandSidePointer->setup( parameters, "right-hand-side-" ) )
   {
      std::cerr << "I was not able to initialize the right-hand side function." << std::endl;
      return false;
   }

   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );

   this->catchExceptions = parameters.getParameter< bool >( "catch-exceptions" );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs() const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( DofVectorPointer& dofVector )
{
   this->uPointer->bind( this->getMesh(), *dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if( MPI::GetSize() > 1 )
   {
      std::cout<<"Nodes Distribution: " << this->distributedMeshPointer->printProcessDistr() << std::endl;
      if( ! Functions::readDistributedMeshFunction( *this->distributedMeshPointer, *this->uPointer, "u", initialConditionFile ) )
      {
         std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
         return false;
      }
      synchronizer.setDistributedGrid( &this->distributedMeshPointer.getData() );
      synchronizer.synchronize( *uPointer );
   }
   else
   {
      if( ! Functions::readMeshFunction( *this->uPointer, "u", initialConditionFile ) )
      {
         std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
         return false;
      }
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename MatrixPointer >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( MatrixPointer& matrixPointer )
{
   const IndexType dofs = this->getDofs();
   typedef typename MatrixPointer::element_type::RowsCapacitiesType RowsCapacitiesTypeType;
   Pointers::SharedPointer<  RowsCapacitiesTypeType > rowLengthsPointer;
   rowLengthsPointer->setSize( dofs );
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, RowsCapacitiesTypeType > matrixSetter;
   matrixSetter.template getCompressedRowLengths< typename Mesh::Cell >(
      this->getMesh(),
      differentialOperatorPointer,
      boundaryConditionPointer,
      rowLengthsPointer );
   matrixPointer->setDimensions( dofs, dofs );
   matrixPointer->setRowCapacities( *rowLengthsPointer );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              DofVectorPointer& dofs )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;

   this->bindDofs( dofs );

   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setIndex( step );

   if( MPI::GetSize() > 1 )
   {
      fileName.setExtension( "pvti" );
      Functions::writeDistributedMeshFunction( *this->distributedMeshPointer, *this->uPointer, "u", fileName.getFileName() );
   }
   else
   {
      fileName.setExtension( "vti" );
      this->uPointer->write( "u", fileName.getFileName() );
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitUpdate( const RealType& time,
                   const RealType& tau,
                   DofVectorPointer& uDofs,
                   DofVectorPointer& fuDofs )
{
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting vectors again if you need.
    */

   this->bindDofs( uDofs );
   this->fuPointer->bind( this->getMesh(), *fuDofs );
   this->explicitUpdater.template update< typename Mesh::Cell >( time, tau, this->getMesh(), this->uPointer, this->fuPointer );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
applyBoundaryConditions( const RealType& time,
                         DofVectorPointer& uDofs )
{
   this->bindDofs( uDofs );
   this->explicitUpdater.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, this->uPointer );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
    template< typename MatrixPointer >
void
HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      DofVectorPointer& dofsPointer,
                      MatrixPointer& matrixPointer,
                      DofVectorPointer& bPointer )
{
   this->bindDofs( dofsPointer );
   this->systemAssembler.template assembly< typename Mesh::Cell, typename MatrixPointer::element_type >(
      time,
      tau,
      this->getMesh(),
      this->uPointer,
      matrixPointer,
      bPointer );
}

} // namespace Problems
} // namespace TNL
