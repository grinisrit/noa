/**
 * \file layers.hh
 * \brief MHFEM solver domain layer indices
 *
 * Implemented by: Gregory Dushkin
 */
#pragma once

namespace noa::test::mhfe {

/// Scope for a domain cell layers indexing enum
struct CellLayers {
	enum : std::size_t {
		/// Computed solution (current step)
		P		    = 0,
		/// Computed solution (previous step)
		P_PREV		= 1,
		/// `a` coefficient
		A		    = 2,
		/// `c` coefficient
		C		    = 3,
		/// Cell measure
		MEASURE		= 4,
		/// Precise solution
		PRECISE		= 5,
		/// `l` coefficient
		L		    = 6,
        /// Scalar function sensitivity with respect to `a`
        CHAIN       = 7,
		/// B<sup>-1</sup> matrix coefficients.
		/// It's an NxN matrix, where N is the number of edges per cell
		/// needing NxN layers to store.
		/// The matrix is stored on layers `[BINV; BINV + N * N)`.
		/// Each consecutive layers stores the next element, so after
		/// BINV there must be a reasonable reserve of latyer indices.
		BINV		= 100,
	};
}; // <-- struct CellLayers

/// Scope for a domain edge layers indexing enum
struct EdgeLayers {
	enum : std::size_t {
		/// Dirichlet border condition
		DIRICHLET	= 0,
		/// Dirichlet border condition mask
		DIRICHLET_MASK	= 100,
		/// Neumann border condition
		NEUMANN		= 1,
		/// Neumann border condition mask
		NEUMANN_MASK	= 101,
		/// Solution over the edges
		TP		= 2,
		/// Sparse system matrix `M` cached row capacities
		ROW_CAPACITIES	= 3,
		/// Sytem right-hand-side vector
		RHS_VECTOR	= 4,
	};
}; // <-- struct EdgeLayers

} // <-- namespace noa::test::mhfe
