#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL {

class HeatEquationBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< HeatEquationBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< HeatEquationBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< HeatEquationBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< HeatEquationBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Please, chose your preferred time discretization  here.
 */
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< HeatEquationBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = false }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< HeatEquationBuildConfigTag, Dimensions >{ enum { enabled = true }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< HeatEquationBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< HeatEquationBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< HeatEquationBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< HeatEquationBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
