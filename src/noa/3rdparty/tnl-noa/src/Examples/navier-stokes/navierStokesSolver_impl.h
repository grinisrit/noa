#ifndef NAVIERSTOKESSOLVER_IMPL_H_
#define NAVIERSTOKESSOLVER_IMPL_H_


#include "navierStokesSolver.h"
#include <stdio.h>
#include <iostream>
#include <TNL/String.h>
#include <TNL/FileName.h>
#include <TNL/Math.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Solvers/ODE/Merson.h>


#ifdef __CUDACC__

#include <cuda.h>

template< typename Real, typename Index >
__device__ void computeVelocityFieldCuda( const Index size,
                                          const Real R,
                                          const Real T,
                                          const Real* rho,
                                          const Real* rho_u1,
                                          const Real* rho_u2,
                                          Real* u1,
                                          Real* u2,
                                          Real* p );
#endif

template< typename Mesh, typename EulerScheme >
navierStokesSolver< Mesh, EulerScheme > :: navierStokesSolver()
: p_0( 0.0 ),
  T( 0.0 ),
  rhsIndex( 0 )
{
}

template< typename Mesh, typename EulerScheme >
   template< typename Geom >
bool navierStokesSolver< Mesh, EulerScheme > :: setMeshGeometry( Geom& geometry ) const
{
   return true;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: setMeshGeometry( tnlLinearGridGeometry< 2, RealType, DeviceType, IndexType >& geometry ) const
{
   geometry. setNumberOfSegments( 3 );
   geometry. setSegmentData( 0, 0.0,  0.0,  1.0 );
   geometry. setSegmentData( 1, 0.5,  0.15, 0.85 );
   geometry. setSegmentData( 2, 1.0,  0.0,  1.0 );
   return true;
}

template< typename Mesh, typename EulerScheme >
   template< typename Real, typename Device, typename Index,
             template< int, typename, typename, typename > class Geometry >
bool navierStokesSolver< Mesh, EulerScheme >::initMesh( Meshes::Grid< 1, Real, Device, Index, Geometry >& mesh,
                                                        const Config::ParameterContainer& parameters ) const
{

}

template< typename Mesh, typename EulerScheme >
   template< typename Real, typename Device, typename Index,
             template< int, typename, typename, typename > class Geometry >
bool navierStokesSolver< Mesh, EulerScheme >::initMesh( Meshes::Grid< 2, Real, Device, Index, Geometry >& mesh,
                                                        const Config::ParameterContainer& parameters ) const
{
   StaticVector< 2, IndexType > meshes;
   meshes.x() = parameters.getParameter< int >( "x-size" );
   meshes.y() = parameters.getParameter< int >( "y-size" );
   if( meshes.x() <= 0 )
   {
      std::cerr << "Error: x-size must be positive integer number! It is " << meshes. x() << " now." << std::endl;
      return false;
   }
   if( meshes.y() <= 0 )
   {
      std::cerr << "Error: y-size must be positive integer number! It is " << meshes. y() << " now." << std::endl;
      return false;
   }
   mesh.setDimensions( meshes. x(), meshes. y() );
   return this->setMeshGeometry( mesh.getGeometry() );
}

template< typename Mesh, typename EulerScheme >
   template< typename Real, typename Device, typename Index,
             template< int, typename, typename, typename > class Geometry >
bool navierStokesSolver< Mesh, EulerScheme >::initMesh( Meshes::Grid< 3, Real, Device, Index, Geometry >& mesh,
                                                        const Config::ParameterContainer& parameters ) const
{

}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme >::setup( const Config::ParameterContainer& parameters )
{
  std::cout << "Initiating solver ... " << std::endl;

   /****
    * Set-up problem type
    */
   const String& problemName = parameters. getParameter< String >( "problem-name" );
   if( problemName == "riser" )
      problem = riser;
   if( problemName == "cavity" )
      problem = cavity;

   /****
    * Set-up the geometry
    */
   const String& meshFile = parameters.getParameter< String >( "mesh" );
   if( ! this->mesh.load( meshFile ) )
   {
      std::cerr << "I am not able to load the mesh from the file " << meshFile << "." << std::endl;
      return false;
   }
   /*StaticVector< 2, RealType > proportions;
   proportions. x() = parameters. getParameter< double >( "width" );
   proportions. y() = parameters. getParameter< double >( "height" );
   if( proportions. x() <= 0 )
   {
      std::cerr << "Error: width must be positive real number! It is " << proportions. x() << " now." << std::endl;
      return false;
   }
   if( proportions. y() <= 0 )
   {
      std::cerr << "Error: height must be positive real number! It is " << proportions. y() << " now." << std::endl;
      return false;
   }
   this->mesh. setOrigin( StaticVector< 2, RealType >( 0, 0 ) );
   this->mesh. setProportions( proportions );

   if( ! this->initMesh( this->mesh, parameters ) )
      return false;
   mesh.refresh();
   mesh.save( String( "mesh.tnl" ) );*/

   nsSolver.setMesh( this->mesh );

   /****
    * Set-up model coefficients
    */
   this->p_0 = parameters. getParameter< double >( "p0" );
   nsSolver.setT( parameters. getParameter< double >( "T") );
   nsSolver.setHeatCapacityRatio( parameters. getParameter< double >( "gamma" ) );
   nsSolver.setMu( parameters. getParameter< double >( "mu") );
   nsSolver.setR( parameters. getParameter< double >( "R") );
   nsSolver.setGravity( parameters. getParameter< double >( "gravity") );
   if( ! this->boundaryConditions.setup( parameters ) )
      return false;

   /****
    * Set-up grid functions
    */
   dofVector. setSize( nsSolver. getDofs() );
   rhsDofVector. setLike( dofVector );
   nsSolver.bindDofVector( dofVector.getData() );

   /****
    * Set-up boundary conditions
    */
   this->boundaryConditions.setMesh( this->mesh );
   nsSolver.setBoundaryConditions( this->boundaryConditions );

   /****
    * Set-up numerical scheme
    */

   pressureGradient.setFunction( nsSolver.getPressure() );
   pressureGradient.bindMesh( this->mesh );
   this->eulerScheme. bindMesh( this->mesh );
   this->eulerScheme. setPressureGradient( this->pressureGradient );
   this->u1Viscosity. bindMesh( this->mesh );
   this->u1Viscosity. setFunction( this->nsSolver.getU1() );
   this->u2Viscosity. bindMesh( this->mesh );
   this->u2Viscosity. setFunction( this->nsSolver.getU2() );
   this->eViscosity. bindMesh( this->mesh );
   this->eViscosity.setFunction( this->nsSolver.getEnergy() );
   nsSolver.setAdvectionScheme( this->eulerScheme );
   nsSolver.setDiffusionScheme( this->u1Viscosity,
                                this->u2Viscosity,
                                this->eViscosity );
   return true;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: setInitialCondition( const Config::ParameterContainer& parameters )
{
   SharedVector< RealType, DeviceType, IndexType > dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e;
   const IndexType& dofs = mesh. getDofs();
   dofs_rho.    bind( & dofVector. getData()[ 0        ], dofs );
   dofs_rho_u1. bind( & dofVector. getData()[     dofs ], dofs );
   dofs_rho_u2. bind( & dofVector. getData()[ 2 * dofs ], dofs );
   dofs_e.      bind( & dofVector. getData()[ 3 * dofs ], dofs );

   dofs_rho. setValue( p_0 / ( this->nsSolver.getR() *
                               this->nsSolver.getT() ) );
   dofs_rho_u1. setValue( 0.0 );
   dofs_rho_u2. setValue( 0.0 );
   dofs_e. setValue( p_0 / ( this->nsSolver.getHeatCapacityRatio() - 1.0 ) );

   const IndexType& xSize = mesh. getDimensions(). x();
   const IndexType& ySize = mesh. getDimensions(). y();
   const RealType hx = mesh. getParametricStep(). x();
   const RealType hy = mesh. getParametricStep(). y();

   for( IndexType j = 0; j < ySize; j ++ )
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c = mesh. getElementIndex( i, j );
         const RealType x = i * hx;
         const RealType y = j * hy;

         dofs_rho. setElement( c, p_0 / ( this->nsSolver.getR() *
                                          this->nsSolver.getT() ) );
         dofs_rho_u1. setElement( c, 0.0 );
         dofs_rho_u2. setElement( c, 0.0 );
         dofs_e. setElement( c, p_0 / ( this->nsSolver.getHeatCapacityRatio() - 1.0 ) );

      }
   return true;
}

template< typename Mesh, typename EulerScheme >
typename navierStokesSolver< Mesh, EulerScheme > :: DofVectorType& navierStokesSolver< Mesh, EulerScheme > :: getDofVector()
{
   return this->dofVector;
}

template< typename Mesh, typename EulerScheme >
bool navierStokesSolver< Mesh, EulerScheme > :: makeSnapshot( const RealType& t,
                                                              const IndexType step )
{
  std::cout << std::endl << "Writing output at time " << t << " step " << step << "." << std::endl;
   if( !nsSolver.writePhysicalVariables( t, step ) )
      return false;
   if( !nsSolver.writeConservativeVariables( t, step ) )
         return false;
   return true;
}

template< typename Mesh, typename EulerScheme >
void navierStokesSolver< Mesh, EulerScheme > :: getExplicitUpdate(  const RealType& time,
                                                                 const RealType& tau,
                                                                 DofVectorType& u,
                                                                 DofVectorType& fu )
{
   nsSolver.getExplicitUpdate( time, tau, u, fu );
   solverMonitor.uMax = this->mesh.getAbsMax( nsSolver.getU() );
   solverMonitor.uAvg = this->mesh.getLpNorm( nsSolver.getU(), 1.0 );
   solverMonitor.eMax = this->mesh.getAbsMax( nsSolver.getEnergy() );
   solverMonitor.eAvg = this->mesh.getLpNorm( nsSolver.getEnergy(), 1.0 );

   /*this->rhsIndex++;
   nsSolver.writePhysicalVariables( time, this->rhsIndex );
   nsSolver.writeConservativeVariables( time, this->rhsIndex );
   nsSolver.writeExplicitRhs( time, this->rhsIndex, fu );
   getchar();*/
}

template< typename Mesh, typename EulerScheme >
SolverMonitor*
   navierStokesSolver< Mesh, EulerScheme > ::  getSolverMonitor()
{
   return &solverMonitor;
}

template< typename Mesh, typename EulerScheme >
String navierStokesSolver< Mesh, EulerScheme > :: getPrologHeader() const
{
   return String( "Navier-Stokes Problem Solver" );
}

template< typename Mesh, typename EulerScheme >
void navierStokesSolver< Mesh, EulerScheme > :: writeProlog( Logger& logger,
                                                             const Config::ParameterContainer& parameters ) const
{
   logger. WriteParameter< String >( "Problem name:", "problem-name", parameters );
   const String& problemName = parameters. getParameter< String >( "problem-name" );
   if( problemName == "cavity" )
   {
      logger. WriteParameter< double >( "Max. inflow velocity:", "max-inflow-velocity", parameters, 1 );
   }
   logger. WriteParameter< double >( "Viscosity:", "mu", parameters );
   logger. WriteParameter< double >( "Temperature:", "T", parameters );
   logger. WriteParameter< double >( "Gass constant:", "R", parameters );
   logger. WriteParameter< double >( "Pressure:", "p0", parameters );
   logger. WriteParameter< double >( "Gravity:", "gravity", parameters );
   logger. WriteSeparator();
   logger. WriteParameter< double >( "Domain width:", mesh. getProportions(). x() - mesh. getOrigin(). x() );
   logger. WriteParameter< double >( "Domain height:", mesh. getProportions(). y() - mesh. getOrigin(). y() );
   logger. WriteSeparator();
   logger. WriteParameter< String >( "Space discretisation:", "scheme", parameters );
   logger. WriteParameter< int >( "Meshes along x:", mesh. getDimensions(). x() );
   logger. WriteParameter< int >( "Meshes along y:", mesh. getDimensions(). y() );
   logger. WriteParameter< double >( "Space step along x:", mesh. getParametricStep(). x() );
   logger. WriteParameter< double >( "Space step along y:", mesh. getParametricStep(). y() );
}

#endif /* NAVIERSTOKESSOLVER_IMPL_H_ */
