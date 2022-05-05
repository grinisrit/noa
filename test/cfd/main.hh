#pragma once

#include <cmath>

// Precise solution to a second set of border conditions
template <typename Real>
inline Real cond2Solution(const Real& x, const Real& y, const Real& t, const Real& a, const Real& c) {
	constexpr auto int_core = [] (const Real& tau, const Real& x, const Real& y, const Real& a) -> Real {
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
		Real integral = 0;
		for (Real tau = from + step/2; tau <= to; tau += step) integral += func(tau, x, y - 5, a) * step;
		// Simple rects
		return integral * sign;
	};
	return (x / sqrt(16 * M_PI * a)) * integrate(int_core, .01, 0, t);
}
