#pragma once

#include <core/mfilename.h>
#include <matrices/tnlMatrixSetter.h>
#include <exception>

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
String HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getType()
{
   return String( "hamiltonJacobiSolver< " ) + Mesh  :: getType() + " >";
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
String HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: getPrologHeader() const
{
   return String( "Hamilton-Jacobi" );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
void HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: writeProlog( tnlLogger& logger,
                                                												 const Config::ParameterContainer& parameters ) const
{
   //logger. WriteParameter< typename String >( "Problem name:", "problem-name", parameters );
   //logger. WriteParameter< String >( "Used scheme:", "scheme", parameters );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide > :: setup( const Config::ParameterContainer& parameters )
{
   this->boundaryCondition.getFunction().setConstant( 1.0 );
   //this->rightHandSide.getFunction().setConstant( 0.0 );
   //return true;
/*
   const String& problemName = parameters. GetParameter< String >( "problem-name" );

   this->schemeTest = parameters. GetParameter< int >( "scheme-test" );
   this->tested = false;

   const String& meshFile = parameters.GetParameter< String >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
	  std::cerr << "I am not able to load the mesh from the file " << meshFile << "." <<std::endl;
	   return false;
   }

   const IndexType& dofs = this->mesh.getDofs();
   dofVector. setSize( dofs );

   this -> u. bind( & dofVector. getData()[ 0 * dofs ], dofs );
   this -> v. bind( & dofVector. getData()[ 1 * dofs ], dofs );

*/
   differentialOperator.getAnisotropy().setConstant( 1.0 ); //setup( parameters );
   differentialOperator.setSmoothing( 1.0 );
   return true;

}


template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
typename HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::IndexType
         HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::getDofs( const MeshType& mesh ) const
{
   /****
    * Set-up DOFs and supporting grid functions
    */
   return mesh.template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename HamiltonJacobi,
          typename BoundaryCondition,
          typename RightHandSide>
void HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
bindDofs( const MeshType& mesh,
          DofVectorType& dofVector )
{
   //const IndexType dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   this->solution.bind( mesh, dofVector );
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshType& mesh,
                     DofVectorType& dofs,
                     MeshDependentDataType& meshDependentData  )
{
  this->bindDofs( mesh, dofs );
  const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
  if( ! Functions::readMeshFunction( this->solution, "u", initialConditionFile ) )
  {
    std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." <<std::endl;
    return false;
  }
  return true;
}

template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
bool
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshType& mesh,
              DofVectorType& dofs,
              MeshDependentDataType& meshDependentData  )
{
  std::cout <<std::endl << "Writing output at time " << time << " step " << step << "." <<std::endl;

   String fileName;
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   this->solution.write( "u", fileName );

   return true;
}


template< typename Mesh,
		  typename HamiltonJacobi,
		  typename BoundaryCondition,
		  typename RightHandSide>
void
HamiltonJacobiProblem< Mesh,HamiltonJacobi,BoundaryCondition,RightHandSide >::
getExplicitUpdate(  const RealType& time,
                 const RealType& tau,
                 const MeshType& mesh,
                 DofVectorType& _u,
                 DofVectorType& _fu,
                 MeshDependentDataType& meshDependentData  )
{

	/*
	if(!(this->schemeTest))
		scheme.getExplicitUpdate(time, tau, _u, _fu);
	else if(!(this->tested))
	{
		this->tested = true;
		DofVectorType tmp;
		if(tmp.setLike(_u))
			tmp = _u;
		scheme.getExplicitUpdate(time, tau, tmp, _u);

	}
	*/

	//this->bindDofs( mesh, _u );
   MeshFunctionType u, fu;
   u.bind( mesh, _u );
   fu.bind( mesh, _fu );
   explicitUpdater.template update< typename MeshType::Cell >( time,
	                                                            mesh,
	                                                            this->differentialOperator,
	                                                            this->boundaryCondition,
	                                                            this->rightHandSide,
	                                                            u,
	                                                            fu );
   //fu.save( "fu.tnl" );
   //std::cerr << "Enter." << std::endl;
   //getchar();
}
