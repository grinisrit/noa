#pragma once

#include "cfd/mhfe/mhfe.hh"

#include <gtest/gtest.h>

template <typename SolverT>
void setupSolver(SolverT& solver) {
	noa::utils::domain::generate2DGrid(solver.getDomain(), 20, 10, 1.0, 1.0);
	solver.updateLayers();

	const auto& mesh = solver.getDomain().getMesh();

	mesh.template forBoundary<SolverT::dimEdge>([&solver, &mesh] (auto edge) {
		const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<SolverT::dimEdge, 0>(edge, 0));
		const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<SolverT::dimEdge, 0>(edge, 1));

		const auto r = p2 - p1;

		if (r[0] == 0) { // x = const
			solver.dirichletMask[edge] = 1;
			solver.dirichlet[edge] = (p1[0] == 0);
			solver.tp[edge] = (p1[0] == 0);
		} else if (r[1] == 0) { // y = const
			solver.neumannMask[edge] = 1;
			solver.neumann[edge] = 0;
		}

		solver.a.forAllElements([] (auto i, auto& v) { v = 1; });
		solver.c.forAllElements([] (auto i, auto& v) { v = 1; });
		solver.cache();
	});
}

template <typename LinearContainer, typename Real = float>
bool TNLIsClose(LinearContainer& c1, LinearContainer& c2, Real relTolerance = 1e-5) {
	Real len1	= Real{};
	Real len2	= Real{};
	Real distance	= Real{};

	c1.forAllElements([&c2, &len1, &len2, &distance] (auto i, auto& v) {
		len1 += v * v;
		len2 += c2[i] * c2[i];

		const auto diff = c2[i] - v;
		distance += diff * diff;
	});

	if (len2 > len1) len1 = len2;

	return (distance / len1) <= relTolerance * relTolerance;
}

/// Verify that identical iniital conditions lead to identical
/// solutions (one step)
#define TEST_MHFEM(method, steps) TEST(method, SolverSteps_ ##steps) { \
	using CellTopology = noa::TNL::Meshes::Topologies::Triangle;\
	constexpr auto features	= noa::test::mhfe::Features{ false, false };\
	using SolverType = noa::test::mhfe::Solver<features, CellTopology>;\
	\
	SolverType s1, s2;\
	\
	setupSolver(s1);\
	setupSolver(s2);\
	\
	for (std::size_t i = 0; i < steps; ++i) {\
		s1.template step<noa::test::mhfe::methods::method>();\
		s2.template step<noa::test::mhfe::methods::method>();\
		\
		TNL_ASSERT_EQ(s1.getM(), s2.getM(), "System matrices are expected to be the same");\
		ASSERT_TRUE(TNLIsClose(s1.p, s2.p)) << " on step " << i << "/"#steps;\
	}\
}

TEST_MHFEM(MHFE, 1);
TEST_MHFEM(MHFE, 100);
TEST_MHFEM(MHFE, 1000);

TEST_MHFEM(LMHFE, 1);
TEST_MHFEM(LMHFE, 100);
TEST_MHFEM(LMHFE, 1000);
