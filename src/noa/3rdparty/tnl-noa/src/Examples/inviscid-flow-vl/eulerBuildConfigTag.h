#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

namespace TNL {

class eulerBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< eulerBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< eulerBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< eulerBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< eulerBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< eulerBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< eulerBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< eulerBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< eulerBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = true }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< eulerBuildConfigTag, Dimensions >{ enum { enabled = ( Dimensions == 1 ) }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< eulerBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< eulerBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< eulerBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< eulerBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL