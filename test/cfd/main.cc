// Greg put your code there

#include <gflags/gflags.h>
#include <iostream>
#include <filesystem>

#include "MHFE.hh"
#include "Func.hh"

// DEFINE_string(mesh, "mesh.vtu", "Path to tetrahedron mesh");
DEFINE_string(outputDir, "./saved", "Directory to output the result to");

using namespace std;
using namespace noa::MHFE::Func;

auto main(int argc, char **argv) -> int {

	gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::cout << FLAGS_outputDir << std::endl;
	if (!std::filesystem::exists(FLAGS_outputDir))
		std::filesystem::create_directory(FLAGS_outputDir);
	if (!std::filesystem::is_directory(FLAGS_outputDir))
		throw std::runtime_error(FLAGS_outputDir + " is not a directory");

	// TODO: get rid of the mess
	using DomainType = noa::MHFE::Storage::Domain<noa::TNL::Meshes::Topologies::Triangle>;
	DomainType domain;

	noa::MHFE::Storage::generate2DGrid(domain, 20, 10, 1, 1);
	// domain.write(FLAGS_mesh);

	noa::MHFE::prepareDomain(domain);

	// Dimesnion constants for 2D
	domain.getMesh().template forBoundary<1>([&] (typename DomainType::GlobalIndexType edge) {
		const auto p1 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<1, 0>(edge, 0));
		const auto p2 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<1, 0>(edge, 1));

		const auto r = p2 - p1;

		if (r[0] == 0) {
			domain.getLayers(1).template get<int>(noa::MHFE::Layer::DIRICHLET_MASK)[edge] = 1;
			domain.getLayers(1).template get<float>(noa::MHFE::Layer::DIRICHLET)[edge] = (p1[0] == 0) * 1;
			domain.getLayers(1).template get<float>(noa::MHFE::Layer::TP)[edge] = (p1[0] == 0) * 1;
		} else if (r[1] == 0) {
			domain.getLayers(1).template get<int>(noa::MHFE::Layer::NEUMANN_MASK)[edge] = 1;
			domain.getLayers(1).template get<float>(noa::MHFE::Layer::NEUMANN)[edge] = 0;
		}
	});

	domain.getLayers(2).template get<float>(noa::MHFE::Layer::A).forAllElements([] (int i, float& v) { v = 1; });
	domain.getLayers(2).template get<float>(noa::MHFE::Layer::C).forAllElements([] (int i, float& v) { v = 1; });

	noa::MHFE::initDomain(domain);

	float T = 10;
	float t = 0;
	float tau = .005;
	do {
		cout << "\r[" << setw(10) << left << t << "/" << right << T << "]";
		cout.flush();
		noa::MHFE::solverStep<MHFEDelta, MHFELumping, MHFERight>(domain, tau);
		t += tau;

		domain.write(FLAGS_outputDir + "/" + std::to_string(t) + ".vtu");
	} while (t <= T);
	cout << endl << "DONE" << endl;

	gflags::ShutDownCommandLineFlags();

	return 0;

}
