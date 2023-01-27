/**
 * \file methods.hh
 * \brief Implements structures specific to MHFE/LMHFE
 *
 * Implemented by: Gregory Dushkin
 */

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>

/// Implements MHFEM method structures
///
/// Every methods structure specialization is required to implement the following
/// static methods:
/// * matrixTerm()
/// * rhs()
namespace noa::test::mhfe::methods {

/// \brief Argument list for method functions
#define __method_args__ \
	Real a, Real c, Real l, Real measure, \
	Real alpha, Real alpha_i, Real beta, Real lambda, \
	Real BinvEl, Real solCell, Real solEdge, \
	Real tau

/// \brief MHFE method functions implementation
/// 
/// Requires specialization for topologies
template <typename CellTopology, typename Real>
struct MHFE {};

template <typename Real>
struct MHFE<TNL::Meshes::Topologies::Triangle, Real> {
	static inline Real delta(__method_args__) {
		return a * (BinvEl - a / l / l / beta);
	} // <-- delta()
	
	static inline Real lumping(__method_args__) {
		return 0;
	} // <-- lumping()
	
	static inline Real rhs(__method_args__) {
		return a * lambda / l / beta * solCell;
	} // <-- rhs()
}; // <-- struct MHFE (Triangle)

/// \brief LMHFE method functions implementation
/// 
/// Requires specialization for topologies
template <typename CellTopology, typename Real>
struct LMHFE {};

template <typename Real>
struct LMHFE<TNL::Meshes::Topologies::Triangle, Real> {
	static inline Real delta(__method_args__) {
		return a * (BinvEl - alpha_i * alpha_i / alpha);
	} // <-- delta()

	static inline Real lumping(__method_args__) {
		return c * measure / 3.0 / tau;
	} // <-- lumping()

	static inline Real rhs(__method_args__) {
		return c * measure * solEdge / 3.0 / tau;
	} // <-- rhs()
}; // <-- struct LMHFE (Triangle)

} // <-- namespace noa::test::mhfe
