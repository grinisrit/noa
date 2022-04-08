#pragma once

namespace noa::MHFE::Func {

using TopoTriangle = TNL::Meshes::Topologies::Triangle;

struct MHFEDelta {
	template <typename CellTopology, typename Real,
		 std::enable_if_t<std::is_same<CellTopology, TopoTriangle>::value, bool> = true>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
			const Real& BinvEl) {
		return a * (BinvEl - a / l / l / beta);
	}

};

struct MHFELumping {
	template <typename CellTopology, typename Real>
	static Real get(const Real& alpha_i, const Real& alpha,
			const Real& lambda, const Real& beta,
			const Real& a, const Real& c, const Real& l,
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
			const Real& prevVal) {
		return a * lambda / l / beta * prevVal;
	}
};

} // <-- namespace noa::MHFE
