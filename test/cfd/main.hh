#pragma once

#include <cmath>

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
