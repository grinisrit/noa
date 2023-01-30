/**
 * \file main.cc
 * \brief Functional tests for the mass lumping technique and adjoint methods in MHFEM (entry point).
 */

// Precompile some headers
#include "main-pch.hh"

// Local headers
#include "mhfe/mhfe.hh"

using noa::utils::test::status_line;

/// Set up GFlags
DEFINE_string(	outputDir,	"./saved",	"Directory to output calculation results to");
DEFINE_bool(	clear,		false,		"Should the output directory be cleared if exists?");
DEFINE_double(	T,		2.0,		"Simulation time");
DEFINE_bool(	lumping,	false,		"Usees LMHFE over MHFE");
DEFINE_int32(	densify,	1,		"Grid density multiplier");
DEFINE_double(	a,		1.0,		"a value");
DEFINE_double(	da,		0.01,		"da value");
DEFINE_bool(	findiff,	false,		"Calculate finite difference");
DEFINE_bool(	adjoint,	false,		"Calculate 'adjoint' sensitivity");

template <typename SolverType>
float solve(SolverType& solver, const noa::utils::Path& solverOutputPath, float av = 1.0f, bool ajoint = false);
template <typename Container, typename IndexType>
auto average(const Container& container, IndexType size) {
	auto sum = container[IndexType{}];
	for (std::size_t i = 1; i < size; ++i) sum += container[i];
	return sum / size;
}

template <typename SolverType>
float fdSensitivityAt(SolverType& solver, float av = 1.0f, float da = 0.001f) {
	constexpr auto dimCell	= SolverType::dimCell;
	using noa::utils::Path;

	solve(solver, Path{}, av);
	const auto av1 = average(solver.p, solver.getDomain().getMesh().template getEntitiesCount<dimCell>());
	solve(solver, Path{}, av + da);
	const auto av2 = average(solver.p, solver.getDomain().getMesh().template getEntitiesCount<dimCell>());

	return (av2 - av1) / da;
}

// Our scalar function
template <typename Real> struct g;

/// Test entry point
int main(int argc, char** argv) {
	// Parse GFlags
	gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	using noa::utils::Path;
	// Process outputDir, perform necessary checks
	const Path outputPath(FLAGS_outputDir);
	std::cout << "Output directory set to: " << outputPath << std::endl;

	using noa::utils::DirectoryStatus;
	switch (noa::utils::check_directory(outputPath)) {
		case DirectoryStatus::NOT_A_DIR:
			std::cerr << "Refusing to continue." << std::endl;
			return EXIT_FAILURE;
		case DirectoryStatus::EXISTS:
			std::cerr << "Output directory exists!" << std::endl;
			if (FLAGS_clear) {
				std::cerr << "Clearing..." << std::endl;
				std::filesystem::remove_all(outputPath);
			} else return EXIT_FAILURE;
		case DirectoryStatus::NOT_FOUND:
			std::filesystem::create_directory(outputPath);
	}

	// outputPath is newly created and guaranteed to be empty
	const auto solverOutputPath = outputPath / "solution";
	std::filesystem::create_directory(solverOutputPath);

	using CellTopology	= noa::TNL::Meshes::Topologies::Triangle;
	constexpr auto features	= noa::test::mhfe::Features{ true, true };
	using SolverType	= noa::test::mhfe::Solver<features, CellTopology>;
	constexpr auto dimCell	= SolverType::DomainType::getMeshDimension();

	SolverType solver;

	if (FLAGS_findiff || FLAGS_adjoint) {
		if (FLAGS_findiff) {
			const auto fdSens = fdSensitivityAt(solver, FLAGS_a, FLAGS_da);
			std::cout << std::endl;
			std::cout << "Finite difference:     " << fdSens << std::endl;
		}

		if (FLAGS_adjoint) {
			const auto ajSens = solve(solver, Path{}, FLAGS_a, true);
			std::cout << std::endl;
			std::cout << "'Adjoint' sensitivity: " << ajSens << std::endl;
		}
	} else {
		solve(solver, solverOutputPath, FLAGS_a);
	}

	return EXIT_SUCCESS;
}

template <typename SolverType>
float solve(SolverType& solver, const noa::utils::Path& solverOutputPath, float av, bool adjoint) {
	solver.reset();

	// Set up the mesh
	noa::utils::domain::generate2DGrid(
			solver.getDomain(),
			20 * FLAGS_densify, 10 * FLAGS_densify,
			1.0 / FLAGS_densify, 1.0 / FLAGS_densify);
	solver.updateLayers();

	// Set up initial & border conditions
	const auto& mesh = solver.getDomain().getMesh();

	mesh.template forBoundary<SolverType::dimEdge>([&solver, &mesh] (auto edge) {
		const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<SolverType::dimEdge, 0>(edge, 0));
		const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<SolverType::dimEdge, 0>(edge, 1));

		const auto r = p2 - p1;

		if (r[0] == 0) { // x = const boundary
			float x1v = 0;

			auto mask = (p1[0] == 0) && (((p1 + r / 2)[1] > 1) && ((p1 + r / 2)[1] < 9));

			solver.dirichletMask[edge] = 1;
			solver.dirichlet[edge] = mask;
			solver.tp[edge] = mask;
		} else if (r[1] == 0) { // y = const boundary
			solver.neumannMask[edge] = 1;
			solver.neumann[edge] = 0;
		}
	});
	solver.a.forAllElements([av] (auto i, auto& v) { v = av; });
	solver.c.forAllElements([] (auto i, auto& v) { v = 1; });
	solver.cache();

	std::ofstream ajsens("ajsens.dat");
	float sensitivity;

	// Solution cycle
	while (solver.t < FLAGS_T) {
		std::cout << status_line << "[" << std::setw(10) << solver.t << " / " << FLAGS_T << "] ["
			<< ((int(solver.t) % 5 == 0) ? "=" : " ")
			<< ((int(solver.t) % 5 == 1) ? "=" : " ")
			<< ((int(solver.t) % 5 == 2) ? "=" : " ")
			<< ((int(solver.t) % 5 == 3) ? "=" : " ")
			<< ((int(solver.t) % 5 == 4) ? "=" : " ") << "]"
			<< " " << sensitivity;

		using namespace noa::test::mhfe::methods;
		if (FLAGS_lumping) {
			solver.template step<LMHFE>();
			if (adjoint) {
				sensitivity = solver.template adjointA<LMHFE, g>();
				ajsens << solver.t << ", " << sensitivity << std::endl;
			}
		} else solver.template step<MHFE>();

		if (!solverOutputPath.empty())
			solver.getDomain().write(
					solverOutputPath / std::filesystem::path(std::to_string(solver.t) + ".vtu"
			));
	}

	return sensitivity;
} // <-- void solve()

template <typename Real>
struct g {
	// g is average P value
	// g = SUM(P_i) / N -> \partial g_P_i = 1 / N
	template <typename TNLVectorView, typename IndexType>
	static inline Real value(TNLVectorView view, IndexType size) {
		return average(view, size);
	}

	template <typename ConstViewType, typename IndexType, typename ViewType>
	static inline void partial_P(ConstViewType view, IndexType size, ViewType output) {
		output.forAllElements([size] (auto idx, auto& v) {
			v = Real{1} / size;
		});
	}

	template <typename ConstViewType, typename IndexType, typename ViewType>
	static inline void partial_A(ConstViewType view, IndexType size, ViewType output) {
		output.forAllElements([] (auto idx, auto& v) { v = Real{}; });
	}
};
