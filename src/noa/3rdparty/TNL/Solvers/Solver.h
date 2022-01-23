// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/TNL/Solvers/SolverInitiator.h>
#include <noa/3rdparty/TNL/Solvers/SolverStarter.h>
#include <noa/3rdparty/TNL/Solvers/SolverConfig.h>
#include <noa/3rdparty/TNL/Config/parseCommandLine.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/MPI/ScopedInitializer.h>
#include <noa/3rdparty/TNL/MPI/Config.h>

namespace noa::TNL {
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

      noa::TNL::MPI::ScopedInitializer mpi( argc, argv );

      if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
         return false;

      SolverInitiator< ProblemSetter, ConfigTag > solverInitiator;
      return solverInitiator.run( parameters );
   }
};

} // namespace Solvers
} // namespace noa::TNL
