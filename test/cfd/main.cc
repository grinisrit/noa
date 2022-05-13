// Greg put your code there

#include <gflags/gflags.h>
#include <iostream>
#include <filesystem>

#include "main.hh"
#include "MHFE.hh"
#include "Func.hh"

// Set up GFlags
DEFINE_bool(clear, false, "Clear the outputDir (WARNING: if outputDir is a file, remove it!)");
DEFINE_bool(precise, false, "Should the saved mesh include precise problem solution");
DEFINE_string(outputDir, "./saved", "Directory to output the result to");
DEFINE_double(T, 10, "Time length of calculations");
DEFINE_int32(condition, 1, "Select boundary condition (1 or 2)");
DEFINE_int32(density, 1, "Grid density multiplier");

using namespace std;
using namespace noa;
using namespace MHFE::Func;

auto main(int argc, char **argv) -> int {
	// Parse GFlags
	gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Process outputDir, perform necessary checks
	cout << "Output directory set to " << FLAGS_outputDir << endl;
	if (std::filesystem::exists(FLAGS_outputDir) && FLAGS_clear) {
		cout << "Clearing output location..." << endl;
		std::filesystem::remove_all(FLAGS_outputDir);
	}
	if (!std::filesystem::exists(FLAGS_outputDir))
		std::filesystem::create_directory(FLAGS_outputDir);
	if (!std::filesystem::is_directory(FLAGS_outputDir))
		throw std::runtime_error(FLAGS_outputDir + " is not a directory");

	using CellTopology	= TNL::Meshes::Topologies::Triangle; // 2D
	using DomainType	= MHFE::Storage::Domain<CellTopology>;
	using GlobalIndex	= typename DomainType::GlobalIndexType;
	constexpr auto cellD	= DomainType::getMeshDimension();	// 2 -> 2D cell
	constexpr auto edgeD	= DomainType::getMeshDimension() - 1;	// 1 -> 2D cell edge
	constexpr auto vertD	= 0;
	DomainType domain;

	// Generate domain grid & prepare it for initialization
	MHFE::Storage::generate2DGrid(domain, 20 * FLAGS_density, 10 * FLAGS_density,
						1.0 / FLAGS_density, 1.0 / FLAGS_density);
	MHFE::prepareDomain(domain, true);

	auto& cellLayers = domain.getLayers(cellD);
	auto& edgeLayers = domain.getLayers(edgeD);
	auto& domainMesh = domain.getMesh();

	// Boundary conditions
	domainMesh.template forBoundary<edgeD>([&] (GlobalIndex edge) {
		const auto p1 = domainMesh.getPoint(domainMesh.template getSubentityIndex<edgeD, vertD>(edge, 0));
		const auto p2 = domainMesh.getPoint(domainMesh.template getSubentityIndex<edgeD, vertD>(edge, 1));

		const auto r = p2 - p1;

		if (r[0] == 0) { // Vertical boundary (x = const)
			edgeLayers.template get<int>(MHFE::Layer::DIRICHLET_MASK)[edge]	= 1;
			// x = 0 -> TP = 1; x = 1 -> TP = 0; x = p1[0]
			float x1v = 0;
			switch (FLAGS_condition) {
				case 1:
					x1v = 1;
					break;
				case 2:
					if (((p1 + r / 2)[1] > 1) && ((p1 + r / 2)[1] < 9)) x1v = 1;
					else x1v = 0;
					break;
			}
			edgeLayers.template get<float>(MHFE::Layer::DIRICHLET)[edge]	= (p1[0] == 0) * x1v;
			edgeLayers.template get<float>(MHFE::Layer::TP)[edge]		= (p1[0] == 0) * x1v;
		} else if (r[1] == 0) { // Horizontal boundary (y = const)
			edgeLayers.template get<int>(MHFE::Layer::NEUMANN_MASK)[edge]	= 1;
			edgeLayers.template get<float>(MHFE::Layer::NEUMANN)[edge]	= 0;
		}
	});

	// Fill a and c coefficients a=1, c=1
	cellLayers.template get<float>(MHFE::Layer::A).forAllElements([] (GlobalIndex i, float& v) { v = 1; });
	cellLayers.template get<float>(MHFE::Layer::C).forAllElements([] (GlobalIndex i, float& v) { v = 1; });

	// Init domain not that the layers are initialized
	MHFE::initDomain(domain);

	// Simulation loop
	float t = 0;		// Current sim time
	float tau = .005;	// Time step
	do {
		cout << "\r[" << setw(10) << left << t << "/" << right << FLAGS_T << "]";
		cout.flush();
		MHFE::solverStep<LMHFEDelta, LMHFELumping, LMHFERight>(domain, tau);
		if (FLAGS_precise) MHFE::writePrecise(domain, cond2Solution<float>, t);
		t += tau;

		domain.write(FLAGS_outputDir + "/" + std::to_string(t) + ".vtu");
	} while (t <= FLAGS_T);
	cout << endl << "DONE" << endl;

	gflags::ShutDownCommandLineFlags();

	return 0;

}
