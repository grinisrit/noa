/**
 * \file main.cc
 * \brief Functional tests for the mass lumping technique and chain methods in MHFEM (entry point).
 */

// Precompile some headers
#include "main-pch.hh"

// Local headers
#include "mhfe/mhfe.hh"

using noa::utils::test::status_line;

/// Set up GFlags
DEFINE_string(	outputDir,	"",		"Directory to output calculation results to. Leave empty to disable");
DEFINE_bool(	clear,		false,		"Should the output directory be cleared if exists?");
DEFINE_double(	T,		2.0,		"Simulation time");
DEFINE_bool(	lumping,	false,		"Usees LMHFE over MHFE");
DEFINE_int32(	densify,	1,		"Grid density multiplier");
DEFINE_double(	a,		1.0,		"a value");
DEFINE_double(	da,		0.01,		"da value");
DEFINE_bool(	findiff,	false,		"Calculate finite difference");
DEFINE_bool(	chain,		false,		"Calculate 'chain rule' sensitivity");

template <typename SolverType>
float solve(SolverType& solver, const noa::utils::Path& solverOutputPath);

template <typename Container, typename IndexType>
auto average(const Container& container, IndexType size) {
	auto sum = container[IndexType{}];
	for (std::size_t i = 1; i < size; ++i) sum += container[i];
	return sum / size;
}

// Our scalar function
template <typename Real> struct g;

/// Test entry point
int main(int argc, char** argv) {
	// Parse GFlags
	gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifdef HAVE_OPENMP
    std::cout << "Have OpenMP" << std::endl;
#else
    std::cout << "No OpenMP" << std::endl;
#endif

	// Check incompatible flags
	if (FLAGS_chain && FLAGS_findiff) {
		std::cerr << "--chain and --findiff are mutually exclusive" << std::endl;
		return EXIT_FAILURE;
	}

	using noa::utils::Path;
	// Process outputDir, perform necessary checks
	const Path outputPath = (FLAGS_outputDir.empty()) ? Path{} : Path{FLAGS_outputDir};
	if (outputPath.empty())
		std::cout << "Won't produce output files. Use --outputDir to change that." << std::endl;
	else {
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
	}

	// outputPath is newly created and guaranteed to be empty
	const auto solverOutputPath = outputPath.empty() ? Path{} : (outputPath / "solution");
	if (!solverOutputPath.empty())
		std::filesystem::create_directory(solverOutputPath);

	using CellTopology	= noa::TNL::Meshes::Topologies::Triangle;
	constexpr auto features	= noa::test::mhfe::Features{ true, true };
	using SolverType	= noa::test::mhfe::Solver<features, CellTopology>;
	constexpr auto dimCell	= SolverType::DomainType::getMeshDimension();

	SolverType solver;

	const auto sensitivity = solve(solver, solverOutputPath);
	std::cout << std::endl;
	if (FLAGS_chain || FLAGS_findiff) {
		std::cout << "Sensitivity: " << sensitivity << std::endl;
	}

	return EXIT_SUCCESS;
}

template <typename SolverType>
float solve(SolverType& solver, const noa::utils::Path& solverOutputPath) {
	solver.reset();

	// Set up the mesh
	auto& domain = solver.getDomain();
	noa::utils::domain::generate2DGrid(
			domain,
			4 * FLAGS_densify, 2 * FLAGS_densify,
			5.0 / FLAGS_densify, 5.0 / FLAGS_densify);

	static constexpr auto solverPostSetup = [] (SolverType& s) {
		s.updateLayers();

		// Set up initial & border conditions
		const auto& mesh = s.getDomain().getMesh();

		mesh.template forBoundary<SolverType::dimEdge>([&s, &mesh] (auto edge) {
			const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<SolverType::dimEdge, 0>(edge, 0));
			const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<SolverType::dimEdge, 0>(edge, 1));

			const auto r = p2 - p1;

			if (r[0] == 0) { // x = const boundary
				float x1v = 0;

				auto mask = (p1[0] == 0) && (((p1 + r / 2)[1] > 1) && ((p1 + r / 2)[1] < 9));

				s.dirichletMask[edge] = 1;
				s.dirichlet[edge] = mask;
				s.tp[edge] = mask;
			} else if (r[1] == 0) { // y = const boundary
				s.neumannMask[edge] = 1;
				s.neumann[edge] = 0;
			}
		});
		s.a.forAllElements([] (auto i, auto& v) { v = FLAGS_a; });
		s.c.forAllElements([] (auto i, auto& v) { v = 1; });
	};

	constexpr auto dimCell	= SolverType::DomainType::getMeshDimension();
	const auto numCells = solver.getDomain().getMesh().template getEntitiesCount<dimCell>();

	std::optional<std::vector<SolverType>> compareSolvers = std::nullopt;
	if (FLAGS_findiff) {
		compareSolvers = std::vector<SolverType>(numCells);
		for (std::size_t i = 0; i < numCells; ++i) {
			auto& solver2 = compareSolvers->at(i);
			solver2 = solver;
			solverPostSetup(solver2);
			solver2.a.forAllElements([idx=i] (auto i, auto& v) { v += FLAGS_da * (i == idx); });
			solver2.cache();
		}
	}

	solverPostSetup(solver);
	solver.cache();

	std::ofstream sensfile("sensfile.dat");
	float sensitivity{};

    using namespace noa::test::mhfe::methods;
    if (FLAGS_chain) {
        solver.template chainAInit<LMHFE>(
            [] (auto view, auto size, auto output) {
                output.forAllElements([size] (auto idx, auto& v) {
                    v = 1.0 / size;
                });
            }
        );
    }

	// Solution cycle
	while (solver.t < FLAGS_T) {
		std::cout << status_line << "[" << std::setw(10) << solver.t << " / " << FLAGS_T << "] ["
			<< ((int(solver.t) % 5 == 0) ? "=" : " ")
			<< ((int(solver.t) % 5 == 1) ? "=" : " ")
			<< ((int(solver.t) % 5 == 2) ? "=" : " ")
			<< ((int(solver.t) % 5 == 3) ? "=" : " ")
			<< ((int(solver.t) % 5 == 4) ? "=" : " ") << "]"
			<< " " << sensitivity;

		if (FLAGS_lumping) {
			solver.template step<LMHFE>();
			if (FLAGS_findiff) for (auto& solver2 : *compareSolvers) solver2.template step<LMHFE>();

			if (FLAGS_chain) {
				sensitivity = solver.template chainA<LMHFE>(
                        [] (auto view, auto size, auto output) {
                            output.forAllElements([size] (auto idx, auto& v) {
                                v = 1.0 / size;
                            });
                        }
                );
			}
		} else {
			solver.template step<MHFE>();
			if (FLAGS_findiff) for (auto& solver2 : *compareSolvers) solver2.template step<MHFE>();
		}

		if (FLAGS_findiff) {
			const auto v1 = average(solver.p, solver.getDomain().getMesh().template getEntitiesCount<dimCell>());
			
			sensitivity = 0;

			for (auto& solver2 : *compareSolvers) {
				const auto v2 = average(solver2.p, solver2.getDomain().getMesh().template getEntitiesCount<dimCell>());
				sensitivity += v2 - v1;
			}

			sensitivity /= FLAGS_da;
		}

		if (FLAGS_chain || FLAGS_findiff) {
			sensfile << solver.t << ", " << sensitivity << std::endl;
		}

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
};
