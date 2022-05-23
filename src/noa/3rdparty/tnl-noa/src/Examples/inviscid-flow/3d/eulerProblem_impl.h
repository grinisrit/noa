#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "LaxFridrichsMomentumZ.h"
#include "EulerPressureGetter.h"
#include "Euler2DVelXGetter.h"
#include "EulerVelGetter.h"

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   return String( "euler3D" );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
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
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( //! this->boundaryConditionPointer->setup( meshPointer, parameters, prefix + "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, prefix + "right-hand-side-" ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs( const MeshPointer& mesh ) const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return 5*mesh->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( const MeshPointer& mesh,
          DofVectorPointer& dofVector )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     const MeshPointer& mesh,
                     DofVectorPointer& dofs,
                     MeshDependentDataPointer& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   gamma = parameters.getParameter< RealType >( "gamma" );
   RealType rhoNWU = parameters.getParameter< RealType >( "NWU-density" );
   RealType velNWUX = parameters.getParameter< RealType >( "NWU-velocityX" );
   RealType velNWUY = parameters.getParameter< RealType >( "NWU-velocityY" );
   RealType velNWUZ = parameters.getParameter< RealType >( "NWU-velocityZ" );
   RealType preNWU = parameters.getParameter< RealType >( "NWU-pressure" );
   //RealType eNWU = ( preNWU / ( rhoNWU * (gamma - 1) ) );
   RealType eNWU = ( preNWU / (gamma - 1) ) + 0.5 * rhoNWU * (std::pow(velNWUX,2)+std::pow(velNWUY,2)+std::pow(velNWUZ,2));
   RealType rhoSWU = parameters.getParameter< RealType >( "SWU-density" );
   RealType velSWUX = parameters.getParameter< RealType >( "SWU-velocityX" );
   RealType velSWUY = parameters.getParameter< RealType >( "SWU-velocityY" );
   RealType velSWUZ = parameters.getParameter< RealType >( "SWU-velocityZ" );
   RealType preSWU = parameters.getParameter< RealType >( "SWU-pressure" );
   //RealType eSWU = ( preSWU / ( rhoSWU * (gamma - 1) ) );
   RealType eSWU = ( preSWU / (gamma - 1) ) + 0.5 * rhoSWU * (std::pow(velSWUX,2)+std::pow(velSWUY,2)+std::pow(velSWUZ,2));
   RealType rhoNWD = parameters.getParameter< RealType >( "NWD-density" );
   RealType velNWDX = parameters.getParameter< RealType >( "NWD-velocityX" );
   RealType velNWDY = parameters.getParameter< RealType >( "NWD-velocityY" );
   RealType velNWDZ = parameters.getParameter< RealType >( "NWD-velocityZ" );
   RealType preNWD = parameters.getParameter< RealType >( "NWD-pressure" );
   //RealType eNWD = ( preNWD / ( rhoNWD * (gamma - 1) ) );
   RealType eNWD = ( preNWD / (gamma - 1) ) + 0.5 * rhoNWD * (std::pow(velNWDX,2)+std::pow(velNWDY,2)+std::pow(velNWDZ,2));
   RealType rhoSWD = parameters.getParameter< RealType >( "SWD-density" );
   RealType velSWDX = parameters.getParameter< RealType >( "SWD-velocityX" );
   RealType velSWDY = parameters.getParameter< RealType >( "SWD-velocityY" );
   RealType velSWDZ = parameters.getParameter< RealType >( "SWD-velocityZ" );
   RealType preSWD = parameters.getParameter< RealType >( "SWD-pressure" );
   //RealType eSWD = ( preSWD / ( rhoSWD * (gamma - 1) ) );
   RealType eSWD = ( preSWD / (gamma - 1) ) + 0.5 * rhoSWD * (std::pow(velSWDX,2)+std::pow(velSWDY,2)+std::pow(velSWDZ,2));
   RealType rhoNEU = parameters.getParameter< RealType >( "NEU-density" );
   RealType velNEUX = parameters.getParameter< RealType >( "NEU-velocityX" );
   RealType velNEUY = parameters.getParameter< RealType >( "NEU-velocityY" );
   RealType velNEUZ = parameters.getParameter< RealType >( "NEU-velocityZ" );
   RealType preNEU = parameters.getParameter< RealType >( "NEU-pressure" );
   //RealType eNEU = ( preNEU / ( rhoNEU * (gamma - 1) ) );
   RealType eNEU = ( preNEU / (gamma - 1) ) + 0.5 * rhoNEU * (std::pow(velNEUX,2)+std::pow(velNEUY,2)+std::pow(velNEUZ,2));
   RealType rhoSEU = parameters.getParameter< RealType >( "SEU-density" );
   RealType velSEUX = parameters.getParameter< RealType >( "SEU-velocityX" );
   RealType velSEUY = parameters.getParameter< RealType >( "SEU-velocityY" );
   RealType velSEUZ = parameters.getParameter< RealType >( "SEU-velocityZ" );
   RealType preSEU = parameters.getParameter< RealType >( "SEU-pressure" );
   //RealType eSEU = ( preSEU / ( rhoSEU * (gamma - 1) ) );
   RealType eSEU = ( preSEU / (gamma - 1) ) + 0.5 * rhoSEU * (std::pow(velSEUX,2)+std::pow(velSEUY,2)+std::pow(velSEUZ,2));
   RealType rhoNED = parameters.getParameter< RealType >( "NED-density" );
   RealType velNEDX = parameters.getParameter< RealType >( "NED-velocityX" );
   RealType velNEDY = parameters.getParameter< RealType >( "NED-velocityY" );
   RealType velNEDZ = parameters.getParameter< RealType >( "NED-velocityZ" );
   RealType preNED = parameters.getParameter< RealType >( "NED-pressure" );
   //RealType eNED = ( preNED / ( rhoNED * (gamma - 1) ) );
   RealType eNED = ( preNED / (gamma - 1) ) + 0.5 * rhoNED * (std::pow(velNEDX,2)+std::pow(velNEDY,2)+std::pow(velNEDZ,2));
   RealType rhoSED = parameters.getParameter< RealType >( "SED-density" );
   RealType velSEDX = parameters.getParameter< RealType >( "SED-velocityX" );
   RealType velSEDY = parameters.getParameter< RealType >( "SED-velocityY" );
   RealType velSEDZ = parameters.getParameter< RealType >( "SED-velocityZ" );
   RealType preSED = parameters.getParameter< RealType >( "SED-pressure" );
   //RealType eSED = ( preSED / ( rhoSED * (gamma - 1) ) );
   RealType eSED = ( preSED / (gamma - 1) ) + 0.5 * rhoSED * (std::pow(velSEDX,2)+std::pow(velSEDY,2)+std::pow(velSEDZ,2));
   RealType x0 = parameters.getParameter< RealType >( "riemann-border" );
   int size = mesh->template getEntitiesCount< Cell >();
   uRho->bind(mesh, dofs, 0);
   uRhoVelocityX->bind(mesh, dofs, size);
   uRhoVelocityY->bind(mesh, dofs, 2*size);
   uRhoVelocityZ->bind(mesh, dofs, 3*size);
   uEnergy->bind(mesh, dofs, 4*size);
   Containers::Vector< RealType, DeviceType, IndexType > data;
   data.setSize(5*size);
   pressure->bind(mesh, data, 0);
   velocity->bind(mesh, data, size);
   velocityX->bind(mesh, data, 2*size);
   velocityY->bind(mesh, data, 3*size);
   velocityZ->bind(mesh, data, 4*size);
   int count = std::pow(size, (1.0/3.0));
   for(IndexType i = 0; i < count; i++)   
      for(IndexType j = 0; j < count; j++)
	 for(IndexType k = 0; k < count; k++)
            if ((i <= x0 * count)&&(j <= x0 * count)&&(k <= x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoSWD;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoSWD * velSWDX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoSWD * velSWDY;
		  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoSWD * velSWDZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eSWD;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velSWDX,2)+std::pow(velSWDY,2)+std::pow(velSWDZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velSWDX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velSWDY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velSWDZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preSWD;
               }
            else
            if ((i <= x0 * count)&&(j <= x0 * count)&&(k > x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoSED;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoSED * velSEDX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoSED * velSEDY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoSED * velSEDZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eSED;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velSEDX,2)+std::pow(velSEDY,2)+std::pow(velSEDZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velSEDX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velSEDY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velSEDZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preSED;
               }
            else
            if ((i <= x0 * count)&&(j > x0 * count)&&(k <= x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoNWD;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoNWD * velNWDX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoNWD * velNWDY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoNWD * velNWDZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eNWD;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velNWDX,2)+std::pow(velNWDY,2)+std::pow(velNWDZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velNWDX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velNWDY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velNWDZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preNWD;
               }
            else
            if ((i <= x0 * count)&&(j > x0 * count)&&(k > x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoNED;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoNED * velNEDX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoNED * velNEDY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoNED * velNEDZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eNED;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velNEDX,2)+std::pow(velNEDY,2)+std::pow(velNEDZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velNEDX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velNEDY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velNEDZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preNED;
               }
            else
            if ((i > x0 * count)&&(j <= x0 * count)&&(k <= x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoSWU;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoSWU * velSWUX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoSWU * velSWUY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoSWU * velSWUZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eSWU;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velSWUX,2)+std::pow(velSWUY,2)+std::pow(velSWUZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velSWUX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velSWUY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velSWUZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preSWU;
               }
            else
            if ((i > x0 * count)&&(j <= x0 * count)&&(k > x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoSEU;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoSEU * velSEUX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoSEU * velSEUY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoSEU * velSEUZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eSEU;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velSEUX,2)+std::pow(velSEUY,2)+std::pow(velSEUZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velSEUX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velSEUY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velSEUZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preSEU;
               }
            else
            if ((i > x0 * count)&&(j > x0 * count)&&(k <= x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoNWU;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoNWU * velNWUX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoNWU * velNWUY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoNWU * velNWUZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eNWU;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velNWUX,2)+std::pow(velNWUY,2)+std::pow(velNWUZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velNWUX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velNWUY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velNWUZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preNWU;
               }
            else
            if ((i > x0 * count)&&(j > x0 * count)&&(k > x0 * count) )
               {
                  (* uRho)[i*std::pow(count,2)+j*count+k] = rhoNEU;
                  (* uRhoVelocityX)[i*std::pow(count,2)+j*count+k] = rhoNEU * velNEUX;
                  (* uRhoVelocityY)[i*std::pow(count,2)+j*count+k] = rhoNEU * velNEUY;
                  (* uRhoVelocityZ)[i*std::pow(count,2)+j*count+k] = rhoNEU * velNEUZ;
                  (* uEnergy)[i*std::pow(count,2)+j*count+k] = eNEU;
                  (* velocity)[i*std::pow(count,2)+j*count+k] = std::sqrt(std::pow(velNEUX,2)+std::pow(velNEUY,2)+std::pow(velNEUZ,2));
                  (* velocityX)[i*std::pow(count,2)+j*count+k] = velNEUX;
                  (* velocityY)[i*std::pow(count,2)+j*count+k] = velNEUY;
                  (* velocityZ)[i*std::pow(count,2)+j*count+k] = velNEUZ;
                  (* pressure)[i*std::pow(count,2)+j*count+k] = preNEU;
               };
   return true; 
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( const MeshPointer& mesh,
                   Matrix& matrix )
{
/*   const IndexType dofs = this->getDofs( mesh );
   typedef typename Matrix::CompressedRowsLengthsVector CompressedRowsLengthsVectorType;
   CompressedRowsLengthsVectorType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, CompressedRowsLengthsVectorType > matrixSetter;
   matrixSetter.template getCompressedRowsLengths< typename Mesh::Cell >( mesh,
                                                                          differentialOperator,
                                                                          boundaryCondition,
                                                                          rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowsLengths( rowLengths ) )
      return false;*/
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              const MeshPointer& mesh,
              DofVectorPointer& dofs,
              MeshDependentDataPointer& meshDependentData )
{
  std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   FileName fileName;
   fileName.setExtension( "tnl" );
   fileName.setIndex( step );
   fileName.setFileNameBase( "rho-" );
   if( ! uRho->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "rhoVelX-" );
   if( ! uRhoVelocityX->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "rhoVelY-" );
   if( ! uRhoVelocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "rhoVelZ-" );
   if( ! uRhoVelocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "energy-" );
   if( ! uEnergy->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocityX-" );
   if( ! velocityX->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocityY-" );
   if( ! velocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocityZ-" );
   if( ! velocityY->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "velocity-" );
   if( ! velocity->save( fileName.getFileName() ) )
      return false;
   fileName.setFileNameBase( "pressue-" );
   if( ! pressure->save( fileName.getFileName() ) )
      return false;

   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitRHS( const RealType& time,
                const RealType& tau,
                const MeshPointer& mesh,
                DofVectorPointer& _u,
                DofVectorPointer& _fu,
                MeshDependentDataPointer& meshDependentData )
{
   typedef typename MeshType::Cell Cell;
   int count = mesh->template getEntitiesCount< Cell >();
   fuRho->bind( mesh, _fu, 0 );
   fuRhoVelocityX->bind( mesh, _fu, count );
   fuRhoVelocityY->bind( mesh, _fu, 2*count );
   fuRhoVelocityZ->bind( mesh, _fu, 3*count );
   fuEnergy->bind( mesh, _fu, 4*count );
   SharedPointer< Continuity > lF3DContinuity;
   SharedPointer< MomentumX > lF3DMomentumX;
   SharedPointer< MomentumY > lF3DMomentumY;
   SharedPointer< MomentumZ > lF3DMomentumZ;
   SharedPointer< Energy > lF3DEnergy;

   this->bindDofs( mesh, _u );
   //rho
   lF3DContinuity->setTau(tau);
   lF3DContinuity->setVelocityX( *velocityX );
   lF3DContinuity->setVelocityY( *velocityY );
   lF3DContinuity->setVelocityZ( *velocityZ );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, Continuity, BoundaryCondition, RightHandSide > explicitUpdaterContinuity; 
   explicitUpdaterContinuity.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF3DContinuity,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRho,
                                                           fuRho );

   //rhoVelocityX
   lF3DMomentumX->setTau(tau);
   lF3DMomentumX->setVelocityX( *velocityX );
   lF3DMomentumX->setVelocityY( *velocityY );
   lF3DMomentumX->setVelocityZ( *velocityZ );
   lF3DMomentumX->setPressure( *pressure );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, MomentumX, BoundaryCondition, RightHandSide > explicitUpdaterMomentumX; 
   explicitUpdaterMomentumX.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF3DMomentumX,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRhoVelocityX,
                                                           fuRhoVelocityX );

   //rhoVelocityY
   lF3DMomentumY->setTau(tau);
   lF3DMomentumY->setVelocityX( *velocityX );
   lF3DMomentumY->setVelocityY( *velocityY );
   lF3DMomentumY->setVelocityZ( *velocityZ );
   lF3DMomentumY->setPressure( *pressure );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, MomentumY, BoundaryCondition, RightHandSide > explicitUpdaterMomentumY;
   explicitUpdaterMomentumY.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF3DMomentumY,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRhoVelocityY,
                                                           fuRhoVelocityY );
 
   //rhoVelocityZ
   lF3DMomentumZ->setTau(tau);
   lF3DMomentumZ->setVelocityX( *velocityX );
   lF3DMomentumZ->setVelocityY( *velocityY );
   lF3DMomentumZ->setVelocityZ( *velocityZ );
   lF3DMomentumZ->setPressure( *pressure );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, MomentumZ, BoundaryCondition, RightHandSide > explicitUpdaterMomentumZ;
   explicitUpdaterMomentumZ.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF3DMomentumZ,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uRhoVelocityZ,
                                                           fuRhoVelocityZ );
  
   //energy
   lF3DEnergy->setTau(tau);
   lF3DEnergy->setVelocityX( *velocityX ); 
   lF3DEnergy->setVelocityY( *velocityY );
   lF3DEnergy->setVelocityZ( *velocityZ );
   lF3DEnergy->setPressure( *pressure );
   Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, Energy, BoundaryCondition, RightHandSide > explicitUpdaterEnergy;
   explicitUpdaterEnergy.template update< typename Mesh::Cell >( time,
                                                           mesh,
                                                           lF3DEnergy,
                                                           this->boundaryConditionPointer,
                                                           this->rightHandSidePointer,
                                                           uEnergy,
                                                           fuEnergy );

/*
cout << "rho " << uRho.getData() << endl;
getchar();
cout << "rhoVelX " << uRhoVelocityX.getData() << endl;
getchar();
cout << "rhoVelY " << uRhoVelocityY.getData() << endl;
getchar();
cout << "Energy " << uEnergy.getData() << endl;
getchar();
cout << "velocity " << velocity.getData() << endl;
getchar();
cout << "velocityX " << velocityX.getData() << endl;
getchar();
cout << "velocityY " << velocityY.getData() << endl;
getchar();
cout << "pressure " << pressure.getData() << endl;
getchar();
*/


/*
   BoundaryConditionsSetter< MeshFunctionType, BoundaryCondition > boundaryConditionsSetter; 
   boundaryConditionsSetter.template apply< typename Mesh::Cell >( 
      this->boundaryCondition, 
      time + tau, 
       u );*/
 }

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
void
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      const MeshPointer& mesh,
                      DofVectorPointer& _u,
                      Matrix& matrix,
                      DofVectorPointer& b,
                      MeshDependentDataPointer& meshDependentData )
{
/*   LinearSystemAssembler< Mesh,
                             MeshFunctionType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             BackwardTimeDiscretisation,
                             Matrix,
                             DofVectorType > systemAssembler;

   MeshFunctionView< Mesh > u( mesh, _u );
   systemAssembler.template assembly< typename Mesh::Cell >( time,
                                                             tau,
                                                             mesh,
                                                             this->differentialOperator,
                                                             this->boundaryCondition,
                                                             this->rightHandSide,
                                                             u,
                                                             matrix,
                                                             b );*/
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
eulerProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshPointer& mesh,
             DofVectorPointer& dofs,
             MeshDependentDataPointer& meshDependentData )
{

   //velocityX
   this->velocityX->setMesh( mesh );
   VelocityX velocityXGetter( *uRho, *uRhoVelocityX );
   *this->velocityX = velocityXGetter;

   //velocityY
   this->velocityY->setMesh( mesh );
   VelocityX velocityYGetter( *uRho, *uRhoVelocityY );
   *this->velocityY = velocityYGetter;

   //velocityY
   this->velocityZ->setMesh( mesh );
   VelocityX velocityZGetter( *uRho, *uRhoVelocityZ );
   *this->velocityZ = velocityZGetter;

   //velocity
   this->velocity->setMesh( mesh );
   Velocity velocityGetter( *uRho, *uRhoVelocityX, *uRhoVelocityY, *uRhoVelocityZ);
   *this->velocity = velocityGetter;

   //pressure
   this->pressure->setMesh( mesh );
   Pressure pressureGetter( *uRho, *uRhoVelocityX, *uRhoVelocityY, *uRhoVelocityZ, *uEnergy, gamma );
   *this->pressure = pressureGetter;

   return true;
}

} // namespace TNL

