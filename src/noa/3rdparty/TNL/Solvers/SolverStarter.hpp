// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Logger.h>
#include <noa/3rdparty/TNL/String.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/MPI/Config.h>
#include <noa/3rdparty/TNL/Solvers/SolverStarter.h>
#include <noa/3rdparty/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/TNL/Solvers/ODE/Merson.h>
#include <noa/3rdparty/TNL/Solvers/ODE/Euler.h>
#include <noa/3rdparty/TNL/Solvers/PDE/ExplicitTimeStepper.h>
#include <noa/3rdparty/TNL/Solvers/PDE/SemiImplicitTimeStepper.h>
#include <noa/3rdparty/TNL/Solvers/PDE/PDESolverTypeResolver.h>

namespace noa::TNL {
namespace Solvers {

template< typename Problem,
          typename ConfigTag,
          bool TimeDependent = Problem::isTimeDependent() >
class TimeDependencyResolver
{};

template< typename Problem,
          typename ConfigTag,
          typename TimeStepper = typename Problem::TimeStepper >
class UserDefinedTimeDiscretisationSetter;

template< typename Problem,
          typename TimeDiscretisation,
          typename ConfigTag,
          bool enabled = ConfigTagTimeDiscretisation< ConfigTag, TimeDiscretisation >::enabled >
class SolverStarterTimeDiscretisationSetter{};

template< typename Problem,
          typename ExplicitSolver,
          typename ConfigTag,
          bool enabled = ConfigTagExplicitSolver< ConfigTag, ExplicitSolver >::enabled >
class SolverStarterExplicitSolverSetter{};


template< typename ConfigTag >
SolverStarter< ConfigTag > :: SolverStarter()
: logWidth( 80 )
{
}

template< typename ConfigTag >
   template< typename Problem >
bool SolverStarter< ConfigTag > :: run( const Config::ParameterContainer& parameters )
{
   /****
    * Create and set-up the problem
    */
   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! MPI::setup( parameters )
    )
      return false;
   Problem problem;
   return TimeDependencyResolver< Problem, ConfigTag >::run( problem, parameters );
}

template< typename Problem,
          typename ConfigTag>
class TimeDependencyResolver< Problem, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         return UserDefinedTimeDiscretisationSetter< Problem, ConfigTag >::run( problem, parameters );
      }
};

template< typename Problem,
          typename ConfigTag>
class TimeDependencyResolver< Problem, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         // TODO: This should be improved - at least rename to LinearSolverSetter
         return SolverStarterTimeDiscretisationSetter< Problem, SemiImplicitTimeDiscretisationTag, ConfigTag, true >::run( problem, parameters );
      }
};

template< typename Problem,
          typename ConfigTag,
          typename TimeStepper >
class UserDefinedTimeDiscretisationSetter
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         TimeStepper timeStepper;
         if( ! timeStepper.setup( parameters ) )
         {
            std::cerr << "The time stepper initiation failed!" << std::endl;
            return false;
         }
         SolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters );
      }
};

template< typename Problem,
          typename ConfigTag >
class UserDefinedTimeDiscretisationSetter< Problem, ConfigTag, void >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         /****
          * Set-up the time discretisation
          */
         const String& timeDiscretisation = parameters.getParameter< String>( "time-discretisation" );
         if( timeDiscretisation == "explicit" )
            return SolverStarterTimeDiscretisationSetter< Problem, ExplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         if( timeDiscretisation == "semi-implicit" )
         {
            if( MPI::GetSize() > 1 )
            {
               std::cerr << "TNL currently does not support semi-implicit solvers with MPI." << std::endl;
               return false;
            }
            return SolverStarterTimeDiscretisationSetter< Problem, SemiImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         }
         if( timeDiscretisation == "implicit" )
         {
            if( MPI::GetSize() > 1 )
            {
               std::cerr << "TNL currently does not support implicit solvers with MPI." << std::endl;
               return false;
            }
            return SolverStarterTimeDiscretisationSetter< Problem, ImplicitTimeDiscretisationTag, ConfigTag >::run( problem, parameters );
         }
         std::cerr << "Uknown time discretisation: " << timeDiscretisation << "." << std::endl;
         return false;
      }
};

/****
 * Setting the time discretisation
 */

template< typename Problem,
          typename TimeDiscretisationTag,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, TimeDiscretisationTag, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         std::cerr << "The time discretisation " << parameters.getParameter< String >( "time-discretisation" ) << " is not supported." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, ExplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         const String& discreteSolver = parameters.getParameter< String>( "discrete-solver" );
         if( discreteSolver != "euler" &&
             discreteSolver != "merson" )
         {
            std::cerr << "Unknown explicit discrete solver " << discreteSolver << ". It can be only: euler or merson." << std::endl;
            return false;
         }
         if( discreteSolver == "euler" )
            return SolverStarterExplicitSolverSetter< Problem, ExplicitEulerSolverTag, ConfigTag >::run( problem, parameters );
         if( discreteSolver == "merson" )
            return SolverStarterExplicitSolverSetter< Problem, ExplicitMersonSolverTag, ConfigTag >::run( problem, parameters );
         return false;
      }
};

template< typename Problem,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, SemiImplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         typedef PDE::SemiImplicitTimeStepper< Problem > TimeStepper;
         SolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters );
      }
};

template< typename Problem,
          typename ConfigTag >
class SolverStarterTimeDiscretisationSetter< Problem, ImplicitTimeDiscretisationTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
//         const String& discreteSolver = parameters.getParameter< String>( "discrete-solver" );
         return false;
      }
};

/****
 * Setting the explicit solver
 */

template< typename Problem,
          typename ExplicitSolverTag,
          typename ConfigTag >
class SolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, false >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         std::cerr << "The explicit solver " << parameters.getParameter< String >( "discrete-solver" ) << " is not supported." << std::endl;
         return false;
      }
};

template< typename Problem,
          typename ExplicitSolverTag,
          typename ConfigTag >
class SolverStarterExplicitSolverSetter< Problem, ExplicitSolverTag, ConfigTag, true >
{
   public:
      static bool run( Problem& problem,
                       const Config::ParameterContainer& parameters )
      {
         typedef PDE::ExplicitTimeStepper< Problem, ExplicitSolverTag::template Template > TimeStepper;
         SolverStarter< ConfigTag > solverStarter;
         return solverStarter.template runPDESolver< Problem, TimeStepper >( problem, parameters );
      }
};

template< typename ConfigTag >
   template< typename Problem,
             typename TimeStepper >
bool SolverStarter< ConfigTag > :: runPDESolver( Problem& problem,
                                                 const Config::ParameterContainer& parameters )
{
   this->totalTimer.reset();
   this->totalTimer.start();

   using SolverMonitorType = IterativeSolverMonitor< typename Problem::RealType,
                                                     typename Problem::IndexType >;
   SolverMonitorType solverMonitor, *solverMonitorPointer( &solverMonitor );

   /****
    * Open the log file
    */
   const String logFileName = parameters.getParameter< String >( "log-file" );
   std::ofstream logFile( logFileName.getString() );
   if( ! logFile ) {
      std::cerr << "Unable to open the log file " << logFileName << "." << std::endl;
      return false;
   }

   /****
    * Set-up the PDE solver
    */
   typename PDE::PDESolverTypeResolver< Problem, TimeStepper >::SolverType solver;
   solver.setComputeTimer( this->computeTimer );
   solver.setIoTimer( this->ioTimer );
   solver.setTotalTimer( this->totalTimer );

   if( problem.getSolverMonitor() )
      solverMonitorPointer = ( SolverMonitorType* ) problem.getSolverMonitor();
   solverMonitorPointer->setVerbose( parameters.getParameter< int >( "verbose" ) );
   solverMonitorPointer->setTimer( this->totalTimer );
   solver.setSolverMonitor( *solverMonitorPointer );

   // catching exceptions ala gtest:
   // https://github.com/google/googletest/blob/59c795ce08be0c8b225bc894f8da6c7954ea5c14/googletest/src/gtest.cc#L2409-L2431
   const int catch_exceptions = parameters.getParameter< bool >( "catch-exceptions" );
   if( catch_exceptions ) {
      try {
         solver.setProblem( problem );
         if( ! solver.setup( parameters ) )
            return false;
      }
      catch ( const std::exception& e ) {
         std::cerr << "Setting up the solver failed due to a C++ exception with description: " << e.what() << std::endl;
         logFile   << "Setting up the solver failed due to a C++ exception with description: " << e.what() << std::endl;
         return false;
      }
      catch (...) {
         std::cerr << "Setting up the solver failed due to an unknown C++ exception." << std::endl;
         logFile   << "Setting up the solver failed due to an unknown C++ exception." << std::endl;
         throw;
      }
   }
   else {
      solver.setProblem( problem );
      //solver.setTimeStepper( timeStepper );
      if( ! solver.setup( parameters ) )
         return false;
   }

   /****
    * Write a prolog
    */
   const int verbose = parameters.getParameter< int >( "verbose" );
   logWidth = parameters.getParameter< int >( "log-width" );
   if( verbose ) {
      Logger logger( logWidth, std::cout );
      solver.writeProlog( logger, parameters );
   }
   Logger logger( logWidth, logFile );
   solver.writeProlog( logger, parameters  );

   /****
    * Set-up timers
    */
   this->computeTimer.reset();
   this->ioTimer.reset();

   /****
    * Create solver monitor thread
    */
   SolverMonitorThread t( solver.getSolverMonitor() );

   /****
    * Start the solver
    */
   bool returnCode = true;
   // catching exceptions ala gtest:
   // https://github.com/google/googletest/blob/59c795ce08be0c8b225bc894f8da6c7954ea5c14/googletest/src/gtest.cc#L2409-L2431
   if( catch_exceptions ) {
      try {
         returnCode = solver.solve();
      }
      catch ( const std::exception& e ) {
         std::cerr << "The solver failed due to a C++ exception with description: " << e.what() << std::endl;
         logFile   << "The solver failed due to a C++ exception with description: " << e.what() << std::endl;
         return false;
      }
      catch (...) {
         std::cerr << "The solver failed due to an unknown C++ exception." << std::endl;
         logFile   << "The solver failed due to an unknown C++ exception." << std::endl;
         throw;
      }
   }
   else {
      returnCode = solver.solve();
   }

   if( ! returnCode ) {
      if( verbose )
         std::cerr << std::endl << "The solver did not converge." << std::endl;
      logFile << "The solver did not converge." << std::endl;
   }

   /****
    * Stop timers
    */
   this->computeTimer.stop();
   this->totalTimer.stop();

   /****
    * Write an epilog
    */
   if( verbose )
      writeEpilog( std::cout, solver );
   writeEpilog( logFile, solver );
   logFile.close();

   return returnCode;
}

template< typename ConfigTag >
   template< typename Solver >
bool SolverStarter< ConfigTag > :: writeEpilog( std::ostream& str, const Solver& solver  )
{
   Logger logger( logWidth, str );
   logger.writeSeparator();
   logger.writeCurrentTime( "Finished at:" );
   if( ! solver.writeEpilog( logger ) )
      return false;
   logger.writeParameter< const char* >( "Compute time:", "" );
   this->computeTimer.writeLog( logger, 1 );
   if( std::is_same< typename Solver::DeviceType, noa::TNL::Devices::Cuda >::value )
   {
      logger.writeParameter< const char* >( "GPU synchronization time:", "" );
      Pointers::getSmartPointersSynchronizationTimer< Devices::Cuda >().writeLog( logger, 1 );
   }
   logger.writeParameter< const char* >( "I/O time:", "" );
   this->ioTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Total time:", "" );
   this->totalTimer.writeLog( logger, 1 );
   char buf[ 256 ];
   sprintf( buf, "%f %%", 100 * ( ( double ) this->totalTimer.getCPUTime() ) / this->totalTimer.getRealTime() );
   logger.writeParameter< char* >( "CPU usage:", buf );
   logger.writeSeparator();
   return true;
}

} // namespace Solvers
} // namespace noa::TNL
