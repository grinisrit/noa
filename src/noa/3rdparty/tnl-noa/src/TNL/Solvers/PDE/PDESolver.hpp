// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/PDESolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Utils.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Real, typename Index >
PDESolver< Real, Index >::PDESolver() : ioTimer( 0 ), computeTimer( 0 ), totalTimer( 0 ), solverMonitorPointer( 0 )
{}

template< typename Real, typename Index >
void
PDESolver< Real, Index >::configSetup( Config::ConfigDescription& config, const String& prefix )
{}

template< typename Real, typename Index >
bool
PDESolver< Real, Index >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   return true;
}

template< typename Real, typename Index >
void
PDESolver< Real, Index >::setSolverMonitor( typename PDESolver< Real, Index >::SolverMonitorType& solverMonitor )
{
   this->solverMonitorPointer = &solverMonitor;
}

template< typename Real, typename Index >
typename PDESolver< Real, Index >::SolverMonitorType&
PDESolver< Real, Index >::getSolverMonitor()
{
   return *this->solverMonitorPointer;
}

template< typename Real, typename Index >
bool
PDESolver< Real, Index >::writeProlog( Logger& logger, const Config::ParameterContainer& parameters )
{
   logger.writeParameter< String >( "Real type:", "real-type", parameters, 0 );
   logger.writeParameter< String >( "Index type:", "index-type", parameters, 0 );
   logger.writeParameter< String >( "Device:", "device", parameters, 0 );
   if( parameters.getParameter< String >( "device" ) == "host" ) {
      if( Devices::Host::isOMPEnabled() ) {
         logger.writeParameter< String >( "OMP enabled:", "yes", 1 );
         logger.writeParameter< int >( "OMP threads:", Devices::Host::getMaxThreadsCount(), 1 );
      }
      else
         logger.writeParameter< String >( "OMP enabled:", "no", 1 );
   }
   if( MPI::isInitialized() )
      logger.writeParameter( "MPI processes:", MPI::GetSize() );
   logger.writeSeparator();
   const bool printGPUs = parameters.getParameter< String >( "device" ) == "cuda";
   logger.writeSystemInformation( printGPUs );
   logger.writeSeparator();
   logger.writeCurrentTime( "Started at:" );
   logger.writeSeparator();
   return true;
}

template< typename Real, typename Index >
void
PDESolver< Real, Index >::setIoTimer( Timer& ioTimer )
{
   this->ioTimer = &ioTimer;
}

template< typename Real, typename Index >
void
PDESolver< Real, Index >::setComputeTimer( Timer& computeTimer )
{
   this->computeTimer = &computeTimer;
}

template< typename Real, typename Index >
void
PDESolver< Real, Index >::setTotalTimer( Timer& totalTimer )
{
   this->totalTimer = &totalTimer;
}

}  // namespace PDE
}  // namespace Solvers
}  // namespace noa::TNL
