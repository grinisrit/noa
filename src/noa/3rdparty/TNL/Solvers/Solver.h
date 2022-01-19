// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Solvers/SolverInitiator.h>
#include <TNL/Solvers/SolverStarter.h>
#include <TNL/Solvers/SolverConfig.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

namespace TNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          template< typename ConfigTag > class ProblemConfig,
          typename ConfigTag = DefaultBuildConfigTag >
struct Solver
{
   static bool run( int argc, char* argv[] )
   {
      Config::ParameterContainer parameters;
      Config::ConfigDescription configDescription;
      ProblemConfig< ConfigTag >::configSetup( configDescription );
      SolverConfig< ConfigTag, ProblemConfig< ConfigTag> >::configSetup( configDescription );
      configDescription.addDelimiter( "Parallelization setup:" );
      Devices::Host::configSetup( configDescription );
      Devices::Cuda::configSetup( configDescription );
      MPI::configSetup( configDescription );

      TNL::MPI::ScopedInitializer mpi( argc, argv );

      if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
         return false;

      SolverInitiator< ProblemSetter, ConfigTag > solverInitiator;
      return solverInitiator.run( parameters );
   }
};

} // namespace Solvers
} // namespace TNL
