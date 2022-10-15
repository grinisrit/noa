#pragma once

#include <cmath>

#include "MHFE.hh"

// Precise solution to a second set of border conditions
template <typename Real>
inline Real cond2Solution(const Real& x, const Real& y, const Real& t, const Real& a, const Real& c) {
	constexpr auto int_core = [] (const Real& tau, const Real& x, const Real& y, const Real& a) -> Real {
		if (tau == 0) {
			if ((x == 0) && (y >= 1) && (y <= 9)) return 1;
			return 0;
		}
		return pow(tau, -1.5)
			* (erf((4 + y) / sqrt(4 * a * tau)) + erf((4 - y) / sqrt(4 * a * tau)))
			* exp(-pow(x / sqrt(4 * a * tau), 2));
	};
	const auto integrate = [&x, &y, &a] (const auto& func, const Real& step, Real from, Real to) -> Real {
		if (to == from) return 0;

		Real sign = 1;
		if (to < from) {
			sign = -1;
			std::swap(from, to);
		}
		// Simpson integration
		Real integral = 0;
		Real delta = 0;
		for (Real tau = from; tau < to; tau += step) {
			delta =	func(tau,		x, y - 5, a);
			delta +=func(tau + step/2,	x, y - 5, a) * 4;
			delta +=func(tau + step,	x, y - 5, a);
			integral += delta * step / 6;
		}
		return integral * sign;
	};
	return (x / sqrt(16 * M_PI * a)) * integrate(int_core, .01, 0, t);
}

// It is a struct because it's passed as a template paramter to testCellSensitivity
struct IntegralOver {
	template <__domain_targs__>
	inline static Real calc(const __DomainType__& domain, const noa::MHFE::Layer& layer) {
		Real sum = 0; // For now just make a sum
		constexpr auto dimCell = domain.getMeshDimension();
		const auto lView = domain.getLayers(dimCell).template get<Real>(layer).getConstView();
		const auto& mesh = domain.getMesh();
		mesh.template forAll<dimCell>([&sum, &mesh, dimCell, lView] (const GlobalIndex& cell) {
			const auto cellE = mesh.template getEntity<dimCell>(cell);
			const Real mes = noa::TNL::Meshes::getEntityMeasure(mesh, cellE);
			sum += lView[cell] * mes;
		});
		return sum;
	}
};
