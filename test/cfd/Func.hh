#pragma once

namespace noa::MHFE::Func {

using TopoTriangle = TNL::Meshes::Topologies::Triangle;

// TODO: Make all the parameters uniform
// TODO: Passing alpha_i assumes that α_i = α_j which may not be true for some topologies

/* ----- MHFE ----- */
struct MHFEDelta {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& BinvEl) {
		return a * (BinvEl - a / l / l / beta);
	}
};

struct MHFELumping {
	template <typename CellTopology, typename Real>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& BinvEl) {
		return 0;
	}
};

struct MHFERight {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& prevVal, const Real& edgeVal) {
		return a * lambda / l / beta * prevVal;
	}
};

/* ----- Lumped MHFE ----- */
struct LMHFEDelta {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& BinvEl) {
		return a * (BinvEl - alpha_i * alpha_i / alpha);
	}
};

struct LMHFELumping {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& BinvEl) {
		return c * mes / 3.0 / tau;
	}
};

struct LMHFERight {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& mes, const Real& tau,
			const Real& prevVal, const Real& edgeVal) {
		return c * mes * edgeVal / 3.0 / tau;
	}
};

} // <-- namespace noa::MHFE
