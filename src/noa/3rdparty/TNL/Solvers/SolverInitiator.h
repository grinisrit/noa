// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Solvers/BuildConfigTags.h>

namespace noaTNL {
namespace Solvers {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename ConfigTag >
class SolverInitiator
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

};

} // namespace Solvers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Solvers/SolverInitiator.hpp>
