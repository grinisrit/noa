// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverInitiator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverStarter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverConfig.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/parseCommandLine.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/ScopedInitializer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Config.h>

namespace noa::TNL {
namespace Solvers {

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   template< typename ConfigTag >
   class ProblemConfig,
   typename ConfigTag = DefaultBuildConfigTag >
struct Solver
{
   static bool
   run( int argc, char* argv[] )
   {
      Config::ParameterContainer parameters;
      Config::ConfigDescription configDescription;
      ProblemConfig< ConfigTag >::configSetup( configDescription );
      SolverConfig< ConfigTag, ProblemConfig< ConfigTag > >::configSetup( configDescription );
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

}  // namespace Solvers
}  // namespace noa::TNL
