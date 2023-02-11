// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace noa::TNL {
namespace Solvers {

class FastBuildConfigTag
{};

/****
 * Turn off support for float and long double.
 */
template<>
struct ConfigTagReal< FastBuildConfigTag, float >
{
   static constexpr bool enabled = false;
};
template<>
struct ConfigTagReal< FastBuildConfigTag, long double >
{
   static constexpr bool enabled = false;
};

/****
 * Turn off support for short int and long int indexing.
 */
template<>
struct ConfigTagIndex< FastBuildConfigTag, short int >
{
   static constexpr bool enabled = false;
};
template<>
struct ConfigTagIndex< FastBuildConfigTag, long int >
{
   static constexpr bool enabled = false;
};

/****
 * Please, chose your preferred time discretisation  here.
 */
template<>
struct ConfigTagTimeDiscretisation< FastBuildConfigTag, ExplicitTimeDiscretisationTag >
{
   static constexpr bool enabled = true;
};
template<>
struct ConfigTagTimeDiscretisation< FastBuildConfigTag, SemiImplicitTimeDiscretisationTag >
{
   static constexpr bool enabled = true;
};
template<>
struct ConfigTagTimeDiscretisation< FastBuildConfigTag, ImplicitTimeDiscretisationTag >
{
   static constexpr bool enabled = false;
};

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
// template<> struct ConfigTagExplicitSolver< FastBuildConfigTag, ExplicitEulerSolverTag >{ static constexpr bool enabled =
// false; };

}  // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<>
struct GridRealTag< Solvers::FastBuildConfigTag, float >
{
   static constexpr bool enabled = false;
};
template<>
struct GridRealTag< Solvers::FastBuildConfigTag, long double >
{
   static constexpr bool enabled = false;
};

/****
 * Turn off support for short int and long int indexing.
 */
template<>
struct GridIndexTag< Solvers::FastBuildConfigTag, short int >
{
   static constexpr bool enabled = false;
};
template<>
struct GridIndexTag< Solvers::FastBuildConfigTag, long int >
{
   static constexpr bool enabled = false;
};

}  // namespace BuildConfigTags
}  // namespace Meshes
}  // namespace noa::TNL
