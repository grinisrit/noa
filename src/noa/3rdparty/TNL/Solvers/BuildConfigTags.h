// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Solvers/ODE/Merson.h>
#include <noa/3rdparty/TNL/Solvers/ODE/Euler.h>

namespace noaTNL {
namespace Solvers {

class DefaultBuildConfigTag {};

/****
 * All devices are enabled by default. Those which are not available
 * are disabled.
 */
template< typename ConfigTag, typename Device > struct ConfigTagDevice{ static constexpr bool enabled = true; };
#ifndef HAVE_CUDA
template< typename ConfigTag > struct ConfigTagDevice< ConfigTag, Devices::Cuda >{ static constexpr bool enabled = false; };
#endif

/****
 * All real types are enabled by default.
 */
template< typename ConfigTag, typename Real > struct ConfigTagReal{ static constexpr bool enabled = true; };

/****
 * All index types are enabled by default.
 */
template< typename ConfigTag, typename Index > struct ConfigTagIndex{ static constexpr bool enabled = true; };

/****
 * The mesh type will be resolved by the Solver by default.
 * (The detailed mesh configuration is in TNL/Meshes/TypeResolver/BuildConfigTags.h)
 */
template< typename ConfigTag > struct ConfigTagMeshResolve{ static constexpr bool enabled = true; };

/****
 * All time discretisations (explicit, semi-impicit and implicit ) are
 * enabled by default.
 */
class ExplicitTimeDiscretisationTag{};
class SemiImplicitTimeDiscretisationTag{};
class ImplicitTimeDiscretisationTag{};

template< typename ConfigTag, typename TimeDiscretisation > struct ConfigTagTimeDiscretisation{ static constexpr bool enabled = true; };

/****
 * All explicit solvers are enabled by default
 */
class ExplicitEulerSolverTag
{
public:
    template< typename Problem, typename SolverMonitor >
    using Template = ODE::Euler< Problem, SolverMonitor >;
};

class ExplicitMersonSolverTag
{
public:
    template< typename Problem, typename SolverMonitor >
    using Template = ODE::Merson< Problem, SolverMonitor >;
};

template< typename ConfigTag, typename ExplicitSolver > struct ConfigTagExplicitSolver{ static constexpr bool enabled = true; };

} // namespace Solvers
} // namespace noaTNL
