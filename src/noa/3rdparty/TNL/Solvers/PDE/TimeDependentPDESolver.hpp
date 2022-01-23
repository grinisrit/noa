// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "TimeDependentPDESolver.h"
#include <noa/3rdparty/TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <noa/3rdparty/TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <noa/3rdparty/TNL/MPI/Wrappers.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
          typename TimeStepper >
TimeDependentPDESolver< Problem, TimeStepper >::
TimeDependentPDESolver()
: problem( 0 ),
  initialTime( 0.0 ),
  finalTime( 0.0 ),
  snapshotPeriod( 0.0 ),
  timeStep( 1.0 )
{
}


template< typename Problem,
          typename TimeStepper >
void
TimeDependentPDESolver< Problem, TimeStepper >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   BaseType::configSetup( config, prefix );
   config.addEntry< String >( prefix + "initial-condition", "File name with the initial condition.", "init.vti" );
   config.addRequiredEntry< double >( prefix + "final-time", "Stop time of the time dependent problem." );
   config.addEntry< double >( prefix + "initial-time", "Initial time of the time dependent problem.", 0 );
   config.addRequiredEntry< double >( prefix + "snapshot-period", "Time period for writing the problem status.");
   config.addEntry< double >( prefix + "time-step", "The time step for the time discretisation.", 1.0 );
   config.addEntry< double >( prefix + "time-step-order", "The time step is set to time-step*pow( space-step, time-step-order).", 0.0 );
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   BaseType::setup( parameters, prefix );

   /////
   // Load the mesh from the mesh file
   //
   const String& meshFile = parameters.getParameter< String >( "mesh" );
   const String& meshFileFormat = parameters.getParameter< String >( "mesh-format" );
   if( MPI::GetSize() > 1 ) {
      if( ! Meshes::loadDistributedMesh( *distributedMeshPointer, meshFile, meshFileFormat ) )
         return false;
      problem->setMesh( distributedMeshPointer );
   }
   else {
      if( ! Meshes::loadMesh( *meshPointer, meshFile, meshFileFormat ) )
         return false;
      problem->setMesh( meshPointer );
   }

   /****
    * Set-up common data
    */
   if( ! this->commonDataPointer->setup( parameters ) )
   {
      std::cerr << "The problem common data initiation failed!" << std::endl;
      return false;
   }
   problem->setCommonData( this->commonDataPointer );

   /****
    * Setup the problem
    */
   if( ! problem->setup( parameters, prefix ) )
   {
      std::cerr << "The problem initiation failed!" << std::endl;
      return false;
   }

   /****
    * Set DOFs (degrees of freedom)
    */
   TNL_ASSERT_GT( problem->getDofs(), 0, "number of DOFs must be positive" );
   this->dofsPointer->setSize( problem->getDofs() );
   this->dofsPointer->setValue( 0.0 );
   this->problem->bindDofs( this->dofsPointer );

   /***
    * Set-up the initial condition
    */
   if( ! this->problem->setInitialCondition( parameters, this->dofsPointer ) ) {
      std::cerr << "Failed to set up the initial condition." << std::endl;
      return false;
   }

   /****
    * Initialize the time discretisation
    */
   bool status = true;
   status &= this->setFinalTime( parameters.getParameter< double >( "final-time" ) );
             this->setInitialTime( parameters.getParameter< double >( "initial-time" ) );
   status &= this->setSnapshotPeriod( parameters.getParameter< double >( "snapshot-period" ) );
   status &= this->setTimeStep( parameters.getParameter< double >( "time-step") );
   status &= this->setTimeStepOrder( parameters.getParameter< double >( "time-step-order" ) );
   if( ! status )
      return false;

   /****
    * Set-up the time stepper
    */
   if( ! this->timeStepper.setup( parameters ) )
      return false;
   this->timeStepper.setSolverMonitor( *this->solverMonitorPointer );
   return true;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
writeProlog( Logger& logger,
             const Config::ParameterContainer& parameters )
{
   logger.writeHeader( problem->getPrologHeader() );
   problem->writeProlog( logger, parameters );
   logger.writeSeparator();
   if( MPI::GetSize() > 1 )
      distributedMeshPointer->writeProlog( logger );
   else
      meshPointer->writeProlog( logger );
   logger.writeSeparator();
   logger.writeParameter< String >( "Time discretisation:", "time-discretisation", parameters );
   if( MPI::GetSize() > 1 )
      logger.writeParameter< double >( "Initial time step:", this->getRefinedTimeStep( distributedMeshPointer->getLocalMesh(), this->timeStep ) );
   else
      logger.writeParameter< double >( "Initial time step:", this->getRefinedTimeStep( *meshPointer, this->timeStep ) );
   logger.writeParameter< double >( "Initial time:", "initial-time", parameters );
   logger.writeParameter< double >( "Final time:", "final-time", parameters );
   logger.writeParameter< double >( "Snapshot period:", "snapshot-period", parameters );
   const String& solverName = parameters. getParameter< String >( "discrete-solver" );
   logger.writeParameter< String >( "Discrete solver:", "discrete-solver", parameters );
   if( solverName == "merson" )
      logger.writeParameter< double >( "Adaptivity:", "merson-adaptivity", parameters, 1 );
   if( solverName == "sor" )
      logger.writeParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solverName == "gmres" ) {
      logger.writeParameter< int >( "Restarting min:", "gmres-restarting-min", parameters, 1 );
      logger.writeParameter< int >( "Restarting max:", "gmres-restarting-max", parameters, 1 );
      logger.writeParameter< int >( "Restarting step min:", "gmres-restarting-step-min", parameters, 1 );
      logger.writeParameter< int >( "Restarting step max:", "gmres-restarting-step-max", parameters, 1 );
   }
   logger.writeParameter< double >( "Convergence residue:", "convergence-residue", parameters );
   logger.writeParameter< double >( "Divergence residue:", "divergence-residue", parameters );
   logger.writeParameter< int >( "Maximal number of iterations:", "max-iterations", parameters );
   logger.writeParameter< int >( "Minimal number of iterations:", "min-iterations", parameters );
   logger.writeSeparator();
   return BaseType::writeProlog( logger, parameters );
}

template< typename Problem,
          typename TimeStepper >
void
TimeDependentPDESolver< Problem, TimeStepper >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
}

template< typename Problem,
          typename TimeStepper >
void
TimeDependentPDESolver< Problem, TimeStepper >::
setInitialTime( const RealType& initialTime )
{
   this->initialTime = initialTime;
}

template< typename Problem,
          typename TimeStepper >
const typename Problem::RealType&
TimeDependentPDESolver< Problem, TimeStepper >::
getInitialTime() const
{
   return this->initialTime;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
setFinalTime( const RealType& finalTime )
{
   if( finalTime <= this->initialTime )
   {
      std::cerr << "Final time for TimeDependentPDESolver must larger than the initial time which is now " << this->initialTime << "." << std::endl;
      return false;
   }
   this->finalTime = finalTime;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename Problem::RealType&
TimeDependentPDESolver< Problem, TimeStepper >::
getFinalTime() const
{
   return this->finalTime;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
setSnapshotPeriod( const RealType& period )
{
   if( period <= 0 )
   {
      std::cerr << "Snapshot tau for TimeDependentPDESolver must be positive value." << std::endl;
      return false;
   }
   this->snapshotPeriod = period;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename Problem::RealType&
TimeDependentPDESolver< Problem, TimeStepper >::
getSnapshotPeriod() const
{
   return this->snapshotPeriod;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0 )
   {
      std::cerr << "The time step for TimeDependentPDESolver must be positive value." << std::endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
}

template< typename Problem,
          typename TimeStepper >
const typename Problem::RealType&
TimeDependentPDESolver< Problem, TimeStepper >::
getTimeStep() const
{
   return this->timeStep;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
solve()
{
   TNL_ASSERT_TRUE( problem, "No problem was set in PDESolver." );

   if( snapshotPeriod == 0 )
   {
      std::cerr << "No snapshot tau was set in TimeDependentPDESolver." << std::endl;
      return false;
   }
   RealType t( this->initialTime );
   IndexType step( 0 );
   IndexType allSteps = ceil( ( this->finalTime - this->initialTime ) / this->snapshotPeriod );

   this->ioTimer->reset();
   this->computeTimer->reset();

   this->ioTimer->start();
   if( ! this->problem->makeSnapshot( t, step, this->dofsPointer ) )
   {
      std::cerr << "Making the snapshot failed." << std::endl;
      return false;
   }
   this->ioTimer->stop();
   this->computeTimer->start();

   /****
    * Initialize the time stepper
    */
   this->timeStepper.setProblem( * ( this->problem ) );
   if( MPI::GetSize() > 1 ) {
      this->timeStepper.init( distributedMeshPointer->getLocalMesh() );
      this->timeStepper.setTimeStep( this->getRefinedTimeStep( distributedMeshPointer->getLocalMesh(), this->timeStep ) );
   }
   else {
      this->timeStepper.init( *meshPointer );
      this->timeStepper.setTimeStep( this->getRefinedTimeStep( *meshPointer, this->timeStep ) );
   }
   while( step < allSteps )
   {
      RealType tau = min( this->snapshotPeriod,
                          this->finalTime - t );
      if( ! this->timeStepper.solve( t, t + tau, this->dofsPointer ) )
         return false;
      step ++;
      t += tau;

      this->ioTimer->start();
      this->computeTimer->stop();
      if( ! this->problem->makeSnapshot( t, step, this->dofsPointer ) )
      {
         std::cerr << "Making the snapshot failed." << std::endl;
         return false;
      }
      this->ioTimer->stop();
      this->computeTimer->start();
   }
   this->computeTimer->stop();

   this->solverMonitorPointer->stopMainLoop();

   return true;
}

template< typename Problem,
          typename TimeStepper >
bool
TimeDependentPDESolver< Problem, TimeStepper >::
writeEpilog( Logger& logger ) const
{
   return ( this->timeStepper.writeEpilog( logger ) &&
      this->problem->writeEpilog( logger ) );
}

} // namespace PDE
} // namespace Solvers
} // namespace noa::TNL
