// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Problems/PDEProblem.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/ExplicitTimeStepper.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/PDE/TimeDependentPDESolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/LinearSolverTypeResolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>

namespace noa::TNL {
namespace Solvers {

template< typename ConfigTag, typename ProblemConfig >
struct SolverConfig
{
   static bool
   configSetup( Config::ConfigDescription& config )
   {
      config.addDelimiter( " === General parameters ==== " );
      config.addEntry< bool >( "catch-exceptions",
                               "Catch C++ exceptions. Disabling it allows the program to drop into the debugger "
                               "and track the origin of the exception.",
                               true );
      /****
       * Setup real type
       */
      config.addEntry< String >( "real-type", "Precision of the floating point arithmetics.", "double" );
      if( ConfigTagReal< ConfigTag, float >::enabled )
         config.addEntryEnum( "float" );
      if( ConfigTagReal< ConfigTag, double >::enabled )
         config.addEntryEnum( "double" );
      if( ConfigTagReal< ConfigTag, long double >::enabled )
         config.addEntryEnum( "long-double" );

      /****
       * Setup device.
       */
      config.addEntry< String >( "device", "Device to use for the computations.", "host" );
      if( ConfigTagDevice< ConfigTag, Devices::Host >::enabled )
         config.addEntryEnum( "host" );
#ifdef HAVE_CUDA
      if( ConfigTagDevice< ConfigTag, Devices::Cuda >::enabled )
         config.addEntryEnum( "cuda" );
#endif

      /****
       * Setup index type.
       */
      config.addEntry< String >( "index-type", "Indexing type for arrays, vectors, matrices etc.", "int" );
      if( ConfigTagIndex< ConfigTag, short int >::enabled )
         config.addEntryEnum( "short-int" );

      if( ConfigTagIndex< ConfigTag, int >::enabled )
         config.addEntryEnum( "int" );

      if( ConfigTagIndex< ConfigTag, long int >::enabled )
         config.addEntryEnum( "long-int" );

      /****
       * Mesh file parameter
       */
      config.addDelimiter( " === Space discretisation parameters ==== " );
      config.addEntry< String >(
         "mesh",
         "A file which contains the numerical mesh. You may create it with tools like tnl-grid-setup or tnl-mesh-convert.",
         "mesh.vti" );
      config.addEntry< String >( "mesh-format", "Mesh file format.", "auto" );

      /****
       * Time discretisation
       */
      config.addDelimiter( " === Time discretisation parameters ==== " );
      using PDEProblem = Problems::PDEProblem< Meshes::Grid< 1, double, Devices::Host, int > >;
      using ExplicitTimeStepper = PDE::ExplicitTimeStepper< PDEProblem, ODE::Euler >;
      PDE::TimeDependentPDESolver< PDEProblem, ExplicitTimeStepper >::configSetup( config );
      ExplicitTimeStepper::configSetup( config );
      if( ConfigTagTimeDiscretisation< ConfigTag, ExplicitTimeDiscretisationTag >::enabled
          || ConfigTagTimeDiscretisation< ConfigTag, SemiImplicitTimeDiscretisationTag >::enabled
          || ConfigTagTimeDiscretisation< ConfigTag, ImplicitTimeDiscretisationTag >::enabled )
      {
         config.addRequiredEntry< String >( "time-discretisation", "Discratisation in time." );
         if( ConfigTagTimeDiscretisation< ConfigTag, ExplicitTimeDiscretisationTag >::enabled )
            config.addEntryEnum( "explicit" );
         if( ConfigTagTimeDiscretisation< ConfigTag, SemiImplicitTimeDiscretisationTag >::enabled )
            config.addEntryEnum( "semi-implicit" );
         if( ConfigTagTimeDiscretisation< ConfigTag, ImplicitTimeDiscretisationTag >::enabled )
            config.addEntryEnum( "implicit" );
      }
      config.addRequiredEntry< String >( "discrete-solver", "The solver of the discretised problem:" );
      if( ConfigTagTimeDiscretisation< ConfigTag, ExplicitTimeDiscretisationTag >::enabled ) {
         if( ConfigTagExplicitSolver< ConfigTag, ExplicitEulerSolverTag >::enabled )
            config.addEntryEnum( "euler" );
         if( ConfigTagExplicitSolver< ConfigTag, ExplicitMersonSolverTag >::enabled )
            config.addEntryEnum( "merson" );
      }
      if( ConfigTagTimeDiscretisation< ConfigTag, SemiImplicitTimeDiscretisationTag >::enabled ) {
         for( auto o : getLinearSolverOptions() )
            config.addEntryEnum( String( o ) );
         config.addEntry< String >( "preconditioner", "The preconditioner for the discrete solver:", "none" );
         for( auto o : getPreconditionerOptions() )
            config.addEntryEnum( String( o ) );
      }
      if( ConfigTagTimeDiscretisation< ConfigTag, ExplicitTimeDiscretisationTag >::enabled
          || ConfigTagTimeDiscretisation< ConfigTag, SemiImplicitTimeDiscretisationTag >::enabled )
      {
         config.addDelimiter( " === Iterative solvers parameters === " );
         IterativeSolver< double, int >::configSetup( config );
      }
      if( ConfigTagTimeDiscretisation< ConfigTag, ExplicitTimeDiscretisationTag >::enabled ) {
         config.addDelimiter( " === Explicit solvers parameters === " );
         ODE::ExplicitSolver< PDEProblem >::configSetup( config );
         if( ConfigTagExplicitSolver< ConfigTag, ExplicitEulerSolverTag >::enabled )
            ODE::Euler< PDEProblem >::configSetup( config );

         if( ConfigTagExplicitSolver< ConfigTag, ExplicitMersonSolverTag >::enabled )
            ODE::Merson< PDEProblem >::configSetup( config );
      }
      if( ConfigTagTimeDiscretisation< ConfigTag, SemiImplicitTimeDiscretisationTag >::enabled ) {
         config.addDelimiter( " === Semi-implicit solvers parameters === " );
         using MatrixType = Matrices::SparseMatrix< double >;
         Linear::CG< MatrixType >::configSetup( config );
         Linear::BICGStab< MatrixType >::configSetup( config );
         Linear::BICGStabL< MatrixType >::configSetup( config );
         Linear::GMRES< MatrixType >::configSetup( config );
         Linear::TFQMR< MatrixType >::configSetup( config );
         Linear::SOR< MatrixType >::configSetup( config );

         Linear::Preconditioners::Diagonal< MatrixType >::configSetup( config );
         Linear::Preconditioners::ILU0< MatrixType >::configSetup( config );
         Linear::Preconditioners::ILUT< MatrixType >::configSetup( config );
      }

      config.addDelimiter( " === Logs and messages ===" );
      config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 2 );
      config.addEntry< String >( "log-file", "Log file for the computation.", "log.txt" );
      config.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
      return true;
   }
};

}  // namespace Solvers
}  // namespace noa::TNL
