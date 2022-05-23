#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL {

class HeatEquationBenchmarkBuildConfigTag{};

namespace Solvers {

/****
 * Turn off support for float and long double.
 */
template<> struct ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< HeatEquationBenchmarkBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< HeatEquationBenchmarkBuildConfigTag, long int >{ enum { enabled = false }; };

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, SemiImplicitTimeDiscretisationTag >{ enum { enabled = false }; };
template<> struct ConfigTagTimeDiscretisation< HeatEquationBenchmarkBuildConfigTag, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, ExplicitEulerSolverTag >{ enum { enabled = true }; };
template<> struct ConfigTagExplicitSolver< HeatEquationBenchmarkBuildConfigTag, ExplicitMersonSolverTag >{ enum { enabled = false }; };

} // namespace Solvers

namespace Meshes {
namespace BuildConfigTags {

template< int Dimensions > struct GridDimensionTag< HeatEquationBenchmarkBuildConfigTag, Dimensions >{ enum { enabled = ( Dimensions == 2 ) }; };

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< HeatEquationBenchmarkBuildConfigTag, float > { enum { enabled = false }; };
template<> struct GridRealTag< HeatEquationBenchmarkBuildConfigTag, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< HeatEquationBenchmarkBuildConfigTag, short int >{ enum { enabled = false }; };
template<> struct GridIndexTag< HeatEquationBenchmarkBuildConfigTag, long int >{ enum { enabled = false }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
