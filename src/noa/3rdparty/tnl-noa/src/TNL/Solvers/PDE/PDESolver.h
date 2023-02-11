// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Logger.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/IterativeSolverMonitor.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Real, typename Index >
class PDESolver
{
public:
   using RealType = Real;
   using IndexType = Index;
   using SolverMonitorType = IterativeSolverMonitor< RealType, IndexType >;

   PDESolver();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   bool
   writeProlog( Logger& logger, const Config::ParameterContainer& parameters );

   void
   setIoTimer( Timer& ioTimer );

   void
   setComputeTimer( Timer& computeTimer );

   void
   setTotalTimer( Timer& totalTimer );

   void
   setSolverMonitor( SolverMonitorType& solverMonitor );

   SolverMonitorType&
   getSolverMonitor();

   bool
   writeEpilog( Logger& logger ) const;

protected:
   Timer *ioTimer, *computeTimer, *totalTimer;

   SolverMonitorType* solverMonitorPointer;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/PDESolver.hpp>
