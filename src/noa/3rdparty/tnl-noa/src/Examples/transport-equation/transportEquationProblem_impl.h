#pragma once

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include <TNL/Exceptions/NotImplementedError.h>

#include "transportEquationProblem.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "Transport Equation" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
   /****
    * Add data you want to have in the computation report (log) as follows:
    * logger.writeParameter< double >( "Parameter description", parameter );
    */
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->velocityField->setup( this->getMesh(), parameters, prefix + "velocity-field-" ) ||
       ! this->differentialOperatorPointer->setup( this->getMesh(), parameters, prefix ) ||
       ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, prefix + "boundary-conditions-" ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs() const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( DofVectorPointer& dofVector )
{
   //const IndexType dofs = this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
   this->uPointer->bind( this->getMesh(), *dofVector );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   if( ! Functions::readMeshFunction( *this->uPointer, "u", initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( Matrix& matrix )
{
   /*const IndexType dofs = this->getDofs();
   typedef typename Matrix::ObjectType::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   Pointers::SharedPointer<  CompressedRowsLengthsVectorType > rowLengths;
   if( ! rowLengths->setSize( dofs ) )
      return false;
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperatorPointer,
                                                                          boundaryConditionPointer,
                                                                          rowLengths );
   matrix->setDimensions( dofs, dofs );
   if( ! matrix->setCompressedRowsLengths( *rowLengths ) )
      return false;*/
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              DofVectorPointer& dofs )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   this->bindDofs( dofs );
   MeshFunctionType printDofs;
   printDofs.bind( this->getMesh(), *dofs );
   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "vti" );
   fileName.setIndex( step );
   printDofs.write( "u", fileName.getFileName() );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitUpdate( const RealType& time,
                   const RealType& tau,
                   DofVectorPointer& _u,
                   DofVectorPointer& _fu )
{
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting mesh dependent data if you need.
    */
   const MeshPointer& mesh = this->getMesh();
   typedef typename MeshType::Cell Cell;
   //int count = ::sqrt(mesh->template getEntitiesCount< Cell >());
   this->bindDofs( _u );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;
   Pointers::SharedPointer<  MeshFunctionType > u;
   u->bind( mesh, *_u );
   Pointers::SharedPointer<  MeshFunctionType > fu;
   fu->bind( mesh, *_fu );
   differentialOperatorPointer->setTau(tau);
   differentialOperatorPointer->setVelocityField( this->velocityField );
   explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   explicitUpdater.template update< typename Mesh::Cell >( time, tau, mesh, u, fu );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
applyBoundaryConditions( const RealType& time,
                         DofVectorPointer& dofs )
{
   this->bindDofs( dofs );
   throw Exceptions::NotImplementedError("TODO: implement BC ... see HeatEquationProblem");
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
void
transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      DofVectorPointer& _u,
                      Matrix& matrix,
                      DofVectorPointer& b )
{
}

} // namespace TNL
