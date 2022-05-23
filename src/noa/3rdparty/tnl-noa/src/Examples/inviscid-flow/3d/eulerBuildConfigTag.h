#ifndef eulerBUILDCONFIGTAG_H_
#define eulerBUILDCONFIGTAG_H_

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

template< int Dimensions > struct ConfigTagDimensions< eulerBuildConfigTag, Dimensions >{ enum { enabled = ( Dimensions == 2 ) }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
template< int Dimensions, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< eulerBuildConfigTag, Meshes::Grid< Dimensions, Real, Device, Index > >
      { enum { enabled = ConfigTagDimensions< eulerBuildConfigTag, Dimensions >::enabled  &&
                         ConfigTagReal< eulerBuildConfigTag, Real >::enabled &&
                         ConfigTagDevice< eulerBuildConfigTag, Device >::enabled &&
                         ConfigTagIndex< eulerBuildConfigTag, Index >::enabled }; };

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
} // namespace TNL


#endif /* eulerBUILDCONFIGTAG_H_ */
