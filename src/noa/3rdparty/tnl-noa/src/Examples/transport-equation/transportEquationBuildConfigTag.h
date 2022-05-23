#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL {

class transportEquationBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< transportEquationBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< transportEquationBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< transportEquationBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< transportEquationBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< transportEquationBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = true }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< transportEquationBuildConfigTag, Solvers::ExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< transportEquationBuildConfigTag, Dimensions >{ enum { enabled = true }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< transportEquationBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< transportEquationBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< transportEquationBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< transportEquationBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
