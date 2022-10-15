#include <noa/kernels.hh> // Instead of compiling alongside kernels.cc

#include <gflags/gflags.h>
#include <iostream>
#include <filesystem>

#include "main.hh"
#include "MHFE.hh"
#include "Func.hh"

// Set up GFlags
DEFINE_bool(clear, false, "Clear the outputDir (WARNING: if outputDir is a file, remove it!)");
DEFINE_bool(precise, false, "Should the saved mesh include precise problem solution");
DEFINE_bool(sensitivity, false, "Perform a sensitibity test");
DEFINE_bool(suite, false, "Perform multiple tests, make an output file");
DEFINE_string(outputDir, "./saved", "Directory to output the result to");
DEFINE_double(T, 10, "Time length of calculations");
DEFINE_double(tau, .005, "Time step for calculations");
DEFINE_double(delta, 0.1, "Sensitivity test delta");
DEFINE_int32(condition, 1, "Select boundary condition (1 or 2)");
DEFINE_int32(density, 1, "Grid density multiplier");
DEFINE_int32(expand, 1, "Multiply mesh dimensions by an integer factor");
DEFINE_bool(lumping, false, "Use mass lumping");

using namespace std;
using namespace noa;
using namespace MHFE::Func;

auto main(int argc, char **argv) -> int {
	// Parse GFlags
	gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Process outputDir, perform necessary checks
	const filesystem::path outputPath(FLAGS_outputDir);
	cout << "Output directory set to " << outputPath << endl;
	if (filesystem::exists(outputPath) && FLAGS_clear) {
		cout << "Clearing output location..." << endl;
		filesystem::remove_all(outputPath);
	}
	constexpr auto checkAndMkdir = [] (const filesystem::path& dirPath) {
		if (!filesystem::exists(dirPath))
			filesystem::create_directory(dirPath);
		else if (!filesystem::is_directory(dirPath))
			throw runtime_error(dirPath.string() + " is not a directory");
	};
	checkAndMkdir(outputPath);

	const auto solverOutputPath = outputPath / "solution";
	checkAndMkdir(solverOutputPath);

	using CellTopology	= TNL::Meshes::Topologies::Triangle; // 2D
	using DomainType	= utils::domain::Domain<CellTopology>;
	using GlobalIndex	= typename DomainType::GlobalIndexType;
	constexpr auto cellD	= DomainType::getMeshDimension();	// 2 -> 2D cell
	constexpr auto edgeD	= DomainType::getMeshDimension() - 1;	// 1 -> 2D cell edge
	constexpr auto vertD	= 0;
	DomainType domain;

	// Generate domain grid & prepare it for initialization
	utils::domain::generate2DGrid(domain, 20 * FLAGS_expand * FLAGS_density, 10 * FLAGS_expand * FLAGS_density,
						1.0 * FLAGS_expand / FLAGS_density, 1.0 * FLAGS_expand / FLAGS_density);
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

	if (FLAGS_sensitivity) {
		float delta = FLAGS_delta;
		ofstream dat;
		if (FLAGS_suite) {
			delta = 0;
			dat.open(outputPath / "sensitivity.dat");
		}
		do {
			cout << "Performing a sensitivity test at t=" << FLAGS_T << " with delta " << delta << endl;
			const auto sens = FLAGS_lumping ?
                                MHFE::testCellSensitivityAt<LMHFEDelta, LMHFELumping, LMHFERight, IntegralOver>(
						domain,
						MHFE::Layer::P, MHFE::Layer::A,
						delta, (float)FLAGS_T) :
                                MHFE::testCellSensitivityAt<MHFEDelta, MHFELumping, MHFERight, IntegralOver>(
						domain,
						MHFE::Layer::P, MHFE::Layer::A,
						delta, (float)FLAGS_T);
			cout << "Solution sensitivity: " << sens << " at delta " << delta << endl;
			if (FLAGS_suite) dat << delta << " " << sens << endl;
			delta += FLAGS_delta / 250;
		} while (delta <= FLAGS_delta);
		if (FLAGS_suite) dat.close();
	} else {
		float tau = FLAGS_tau;	// Time step
		float T = FLAGS_T;
		// Save the step result
		const MHFE::PostStepCb<float> saveStep = [&domain, &tau, &solverOutputPath] (const float& t) -> void {
			if (FLAGS_precise) MHFE::writePrecise(domain, cond2Solution<float>, t - tau);
			domain.write(solverOutputPath / filesystem::path(to_string(t) + ".vtu"));
		};
		if (FLAGS_lumping)
			MHFE::simulateTo<LMHFEDelta, LMHFELumping, LMHFERight>(domain, T, tau, saveStep);
		else
			MHFE::simulateTo<MHFEDelta, MHFELumping, MHFERight>(domain, T, tau, saveStep);
	}

	gflags::ShutDownCommandLineFlags();

	return 0;

}
