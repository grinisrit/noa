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

	SolverType solver;
	noa::utils::domain::generate2DGrid(
			solver.getDomain(),
			20 * FLAGS_densify, 10 * FLAGS_densify,
			1.0 / FLAGS_densify, 1.0 / FLAGS_densify);
	solver.updateLayers();
	
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
	solver.a.forAllElements([] (auto i, auto& v) { v = 1; });
	solver.c.forAllElements([] (auto i, auto& v) { v = 1; });
	solver.cache();

	while (solver.t < FLAGS_T) {
		std::cout << status_line << "[" << std::setw(10) << solver.t << " / " << FLAGS_T << "] ["
			<< ((int(solver.t) % 5 == 0) ? "=" : " ")
			<< ((int(solver.t) % 5 == 1) ? "=" : " ")
			<< ((int(solver.t) % 5 == 2) ? "=" : " ")
			<< ((int(solver.t) % 5 == 3) ? "=" : " ")
			<< ((int(solver.t) % 5 == 4) ? "=" : " ") << "]";
		if (FLAGS_lumping)
			solver.template step<noa::test::mhfe::methods::LMHFE>();
		else
			solver.template step<noa::test::mhfe::methods::MHFE>();

		solver.getDomain().write(solverOutputPath / std::filesystem::path(std::to_string(solver.t) + ".vtu"));
	}
	std::cout << std::endl;

	return EXIT_SUCCESS;
}
