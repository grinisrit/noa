/**
 * \file torchext.cc
 * \brief Torch extension for MHFEM
 */

// Standard library
#include <functional>
#include <memory>
#include <optional>

// Torch/pybind
#include <torch/extension.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>

// TinyXML
#include <noa/kernels.hh>

// NOA headers
#include <noa/utils/common.hh>
// Test NOA headers
#include <mhfe/mhfe.hh>

// Local headers
#include "dtype.hh"
#include "wrappers.hh"
using namespace noa::test::utils::pytorch;

// Compile-time options setup
#ifndef REAL
#define REAL float
#endif
using Real = REAL;

#ifndef TOPOLOGY
#define TOPOLOGY noa::TNL::Meshes::Topologies::Triangle
#endif
using CellTopology = TOPOLOGY;

#ifndef PRECISE
#define PRECISE false
#endif
static constexpr bool usePrecise = PRECISE;

#ifndef CALCULATE_CHAIN
#define CALCULATE_CHAIN false
#endif
static constexpr bool calculateChain = CALCULATE_CHAIN;

static constexpr auto solverFeatures = noa::test::mhfe::Features{ usePrecise, calculateChain };

using Solver = noa::test::mhfe::Solver<solverFeatures, CellTopology, noa::TNL::Devices::Host, Real>;

#ifdef HAVE_OPENMP
#warning "Youre not supposed to have OpenMP ON!"
#endif

/// A "hello" function
inline void hi() {
#define NOA_STR_INT(X) #X
#define NOA_STR(X) NOA_STR_INT(X)
	std::cout << "Hello from NOA MHFEM solver module!" << std::endl;
	std::cout << "Info about this module:" << std::endl;
	std::cout
		<< "\tReal type:                       " NOA_STR(REAL) << std::endl
		<< "\tCell topology:                   " NOA_STR(TOPOLOGY) << std::endl
		<< "\tWrite precise solution:          " NOA_STR(PRECISE) << std::endl
		<< "\tCalculate chain rule derivative: " NOA_STR(CALCULATE_CHAIN) << std::endl
		;
#undef NOA_STR
#undef NOA_STR_INT
}


/// \brief Wrap domain layer data into a torch tensor
template <std::size_t dimension, template <std::size_t> class TDTFetch>
constexpr auto layer2Tensor = [] (Solver& solver, auto& layer) -> torch::Tensor {
    if constexpr (std::is_same_v<std::remove_reference_t<decltype(layer)>, noa::utils::test::Empty>) {
        return torch::tensor({});
    } else {
        const auto size = solver.getDomain().getMesh().template getEntitiesCount<dimension>();
        using ValueType = decltype(layer[0]);
        const auto options = torch::TensorOptions().dtype(TDTFetch<sizeof(ValueType)>::value);
        return torch::from_blob(layer.getData(), size, options);
    }
};

/// \brief Get member function-pointer for realtime dimension template parameter value
///
/// TNL uses template arguments to pass dimensions to functions. Whwen binding to Python,
/// we want to be able to recieve them at runtime. This macro generates utility functions
/// for this
#define rtFP(name, method) \
    template <std::size_t dimension = Solver::dimCell>\
    auto name(std::size_t rtDimension) {\
        if (rtDimension > Solver::dimCell) {\
            throw std::runtime_error("Requested dimension higher than cell dimension!");\
        }\
        if (rtDimension == dimension) {\
            return &method <dimension>;\
        }\
        if constexpr (dimension == 0) {\
            throw std::runtime_error("dimension==0 past return. Unreachable!");\
        } else {\
            return name<dimension - 1>(rtDimension);\
        }\
    }

rtFP(rtGetEntitiesCount, Solver::DomainType::MeshType::getEntitiesCount);

/// \brief Get entity points count
template <std::size_t dimension = Solver::dimCell>
auto getPointsCount(const Solver::DomainType::MeshType& mesh, std::size_t rtDimension, std::size_t entityIndex) {
    if (rtDimension > Solver::dimCell) {
        throw std::runtime_error("Requested dimension higher than cell dimension!");
    }
    if (rtDimension == dimension) {
        return mesh.template getSubentitiesCount<dimension, 0>(entityIndex);
    }

    if constexpr (dimension <= 1) {
        throw std::runtime_error("dimension<=1 past getSubentitiesCount call. Unreachable!");
    } else {
        return getPointsCount<dimension - 1>(mesh, rtDimension, entityIndex);
    }
} // <-- getPointsCount()

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	// Export TNL Mesh interface
	using MeshWrapper = ConstWeakWrapper<Solver::DomainType::MeshType>;
    using EntityIteratorCb = void(std::size_t, const torch::Tensor&);

	py::class_<MeshWrapper>(m, "Mesh")
		.def(
			"getEntitiesCount",
			[] (MeshWrapper& mesh, std::size_t dimension) -> std::size_t {
				return std::invoke(rtGetEntitiesCount(dimension), mesh.get());
			}
		)
		.def(
			"forBoundary",
			[] (MeshWrapper& mesh, const std::function<void(std::size_t, const torch::Tensor&, const torch::Tensor&)>& f) -> void {
				mesh->forBoundary<Solver::dimEdge>([&f, mesh] (auto edge) {
					const auto p1 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimEdge, 0>(edge, 0));
					const auto p2 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimEdge, 0>(edge, 1));
					f(edge, torch::tensor({p1[0], p1[1]}), torch::tensor({p2[0], p2[1]}));
				});
			}
		)
        .def(
            "forAllPoints",
            [] (MeshWrapper& mesh, const std::function<void(std::size_t, const torch::Tensor&)> f) -> void {
                mesh->forAll<0>([&f, mesh] (auto point) {
                        const auto p = mesh->getPoint(point);
                        f(point, torch::tensor({p[0], p[1]}));
                });
            }
        )
        .def(
            "forAllEdges",
            [] (MeshWrapper& mesh, const std::function<void(std::size_t, const torch::Tensor&, const torch::Tensor&)>& f) -> void {
                mesh->forAll<Solver::dimEdge>([&f, mesh] (auto edge) {
					const auto p1 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimEdge, 0>(edge, 0));
					const auto p2 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimEdge, 0>(edge, 1));
					f(edge, torch::tensor({p1[0], p1[1]}), torch::tensor({p2[0], p2[1]}));
                });
            }
        )
        .def(
            "forAllCells",
            [] (MeshWrapper& mesh, const std::function<void(std::size_t, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>& f) -> void {
                mesh->forAll<Solver::dimCell>([&f, mesh] (auto cell) {
					const auto p1 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimCell, 0>(cell, 0));
					const auto p2 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimCell, 0>(cell, 1));
					const auto p3 = mesh->getPoint(mesh->template getSubentityIndex<Solver::dimCell, 0>(cell, 2));
					f(cell, torch::tensor({p1[0], p1[1]}), torch::tensor({p2[0], p2[1]}), torch::tensor({p3[0], p3[1]}));
                });
            }
        )
		;

	// Export Domain interface
	using DomainWrapper = WeakWrapper<Solver::DomainType>;
	py::class_<DomainWrapper>(m, "Domain")
		.def(
			"getMesh",
			[] (DomainWrapper& domain) -> std::optional<MeshWrapper> {
				if (domain->isClean()) return std::nullopt;
				return MeshWrapper{&domain->getMesh()};
			}
		)
		.def(
			"loadFrom",
			[] (
				DomainWrapper& domain,
				const std::string& filename,
				const std::map<std::string, std::size_t>& layersRequest
			) -> void {
				if (!domain->isClean()) domain->clear();
				domain->loadFrom(filename, layersRequest);
			}
		)
		.def(
			"write",
			[] (DomainWrapper& domain, const std::string_view path) -> void {
				domain->write(path);
			}
		)
        .def(
            "cat",
            [] (DomainWrapper& domain) -> std::string {
                std::stringstream ss;
                domain->writeStream(ss);
                return ss.str();
            }
        )
        .def(
            "vistify",
            [] (DomainWrapper& domain) {
                // Import Python modules
                const auto pyvista = py::module_::import("pyvista");
                const auto tempfile = py::module_::import("tempfile");

                // Create a temp file
                using namespace pybind11::literals;
                const auto tFile = tempfile.attr("NamedTemporaryFile")("suffix"_a = ".vtu");
                const std::string file = tFile.attr("name").template cast<std::string>();
                // std::cout << "Temp file name: " << file << std::endl;

                // Save the domain to a temp file
                domain->write(file);
                
                // Load the file with PyVista
                return pyvista.attr("read")(file);
            }
        )
		;

	// Export solver interface
	py::class_<Solver>(m, "Solver")
		.def(py::init<>())
		.def("reset", &Solver::reset)
		.def("updateLayers", &Solver::updateLayers)
		.def("getDomain", [] (Solver& solver) { return DomainWrapper{&solver.getDomain()}; })
        .def("getTau", [] (Solver& solver) { return solver.tau; })
        .def("setTau", [] (Solver& solver, Real tau) { solver.tau = tau; })
        .def("getT", [] (Solver& solver) { return solver.t; })
        .def(
            "getLayer",
            [] (Solver& solver, const std::string& layer) -> torch::Tensor {
                return std::unordered_map<std::string, torch::Tensor>{
                    { "neumann",       layer2Tensor<Solver::dimEdge, TDTReal>(solver, solver.neumann)       },
                    { "neumannMask",   layer2Tensor<Solver::dimEdge, TDTInt >(solver, solver.neumannMask)   },
                    { "dirichlet",     layer2Tensor<Solver::dimEdge, TDTReal>(solver, solver.dirichlet)     },
                    { "dirichletMask", layer2Tensor<Solver::dimEdge, TDTInt >(solver, solver.dirichletMask) },
                    { "edgeSolution",  layer2Tensor<Solver::dimEdge, TDTReal>(solver, solver.tp)            },

                    { "solution",      layer2Tensor<Solver::dimCell, TDTReal>(solver, solver.p)             },
                    { "a",             layer2Tensor<Solver::dimCell, TDTReal>(solver, solver.a)             },
                    { "c",             layer2Tensor<Solver::dimCell, TDTReal>(solver, solver.c)             },

                    { "chainA",        layer2Tensor<Solver::dimCell, TDTReal>(solver, solver.scalarWrtA)    },
                }.at(layer);
                // return layer2Tensor<Solver::dimEdge, TDTReal>(solver, solver.neumann);
            }
        )
        .def("cache", &Solver::cache)
        .def(
            "step",
            [] (Solver& solver, const std::string& method) -> void {
                if (method == "mhfe") {
                    solver.template step<noa::test::mhfe::methods::MHFE>();
                } else if (method == "lmhfe") {
                    solver.template step<noa::test::mhfe::methods::LMHFE>();
                } else {
                    throw std::runtime_error("Unknown method!");
                }
            }
        )
        .def(
            "chainInit",
            [] (Solver& solver, const std::string& method, const std::function<torch::Tensor(Solver& solver)>& fWrtP) {
                using namespace noa::test::mhfe::methods;
                if (method == "lmhfe") {
                    solver.template chainAInit<LMHFE>(fWrtP);
                } else if (method == "mhfe") {
                    solver.template chainAInit<MHFE>(fWrtP);
                } else {
                    throw std::runtime_error("Unknown method!");
                }
            }
        )
        .def(
            "chainStep",
            [] (Solver& solver, const std::string& method, const std::function<torch::Tensor(Solver& solver)>& fWrtP) -> void {
                const auto fWrtP_ = [&fWrtP, &solver] (auto view, auto size, auto output) {
                    const auto result = fWrtP(solver).contiguous();
                    if (result.sizes()[0] != size) {
                        throw std::runtime_error(
                                "Returned tensor size " + std::to_string(result.sizes()[0])
                                + " != " + std::to_string(size)
                        );
                    }
                    const auto data = result.data_ptr<Real>();
                    output.forAllElements(
                            [&data] (auto index, auto& value) { value = data[index]; }
                    );
                };

                if (method == "mhfe") {
                    solver.template chainA<noa::test::mhfe::methods::MHFE>(fWrtP_);
                } else if (method == "lmhfe") {
                    solver.template chainA<noa::test::mhfe::methods::LMHFE>(fWrtP_);
                } else {
                    throw std::runtime_error("Unknown method!");
                }
            }
        )
		;

	m.def("hi", &hi, py::call_guard<py::gil_scoped_release>(), "Say hi and print the module compile-time features");

    m.def(
        "generate2DRect",
        [] (DomainWrapper& domain, std::size_t Nx, std::size_t Ny, Real dx, Real dy) -> void {
            if (!domain->isClean()) domain->clear();
            noa::utils::domain::generate2DGrid(domain.get(), Nx, Ny, dx, dy);
        }
    );
    m.def(
        "generate2DRectOffset",
        [] (DomainWrapper& domain, std::size_t Nx, std::size_t Ny, Real dx, Real dy, Real offsetX, Real offsetY) -> void {
            if (!domain->isClean()) domain->clear();
            noa::utils::domain::generate2DGrid(domain.get(), Nx, Ny, dx, dy, offsetX, offsetY);
        }
    );

    static constexpr auto initSolver = [] (Solver& solver, int Nx, int Ny) {
        noa::utils::domain::generate2DGrid(
            solver.getDomain(),
            Nx, Ny,
            20.0 / Nx, 10.0 / Ny
        );

        solver.updateLayers();

        // Set up initial & border conditions
        const auto& mesh = solver.getDomain().getMesh();

        mesh.template forBoundary<Solver::dimEdge>([&solver, &mesh] (auto edge) {
            const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<Solver::dimEdge, 0>(edge, 0));
            const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<Solver::dimEdge, 0>(edge, 1));

            const auto r = p2 - p1;

            if (r[0] == 0) { // x = const boundary
                float x1v = 0;

                const bool mask = (p1[0] == 0) && (((p1 + r / 2)[1] > 4) && ((p1 + r / 2)[1] < 6));

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
    };

    m.def(
        "benchmarkChain",
        [] (std::size_t steps) -> Real {
            Solver solver;

            initSolver(solver, 40, 20);
            solver.cache();

            static constexpr auto gWrtP = [] (auto view, auto size, auto output) {
                output.forAllElements(
                    [size] (auto, auto& v) {
                        v = Real{1} / size;
                    }
                );
            };

            const auto cells = solver.getDomain().getMesh().template getEntitiesCount<Solver::dimCell>();

            using namespace noa::test::mhfe::methods;
            solver.template chainAInit<LMHFE>(gWrtP);

            for (std::size_t step = 0; step < steps; ++step) {
                solver.template step<LMHFE>();
                solver.template chainA<LMHFE>(gWrtP);
            }

            Real ret = 0;
            for (std::size_t cell = 0; cell < cells; ++cell) ret += solver.scalarWrtA[cell];
            return ret;
        }
    );
    m.def(
        "benchmarkFindiff",
        [] (std::size_t steps, Real da = 0.01) -> Real {
            Solver baseSolver;

            initSolver(baseSolver, 40, 20);
            baseSolver.cache();

            const auto cells = baseSolver.getDomain().getMesh().template getEntitiesCount<Solver::dimCell>();

            std::vector<Solver> solvers(cells);
            for (std::size_t cell = 0; cell < cells; ++cell) {
                auto& s = solvers.at(cell);
                initSolver(s, 40, 20);
                s.a[cell] += da;
                s.cache();
            }

            for (std::size_t step = 0; step < steps; ++step) {
                using namespace noa::test::mhfe::methods;
                baseSolver.template step<LMHFE>();
                for (auto&& s : solvers) s.template step<LMHFE>();
            }

            Real ret = 0;
            for (const auto& s : solvers) {
                for (std::size_t cell = 0; cell < cells; ++cell) ret += s.p[cell] - baseSolver.p[cell];
            }
            return ret / (cells * da);
        }
    );

    m.def(
        "statChain",
        [] (std::size_t steps) -> torch::Tensor {
            Solver solver;

            initSolver(solver, 10, 5);
            solver.cache();

            static constexpr auto gWrtP = [] (auto view, auto size, auto output) {
                output.forAllElements(
                    [size] (auto, auto& v) {
                        v = Real{1} / size;
                    }
                );
            };

            const auto cells = solver.getDomain().getMesh().template getEntitiesCount<Solver::dimCell>();

            using namespace noa::test::mhfe::methods;
            solver.template chainAInit<LMHFE>(gWrtP);

            const auto options = torch::TensorOptions().dtype(tdtReal<Real>);
            auto ret = torch::zeros(steps, options);

            for (std::size_t step = 0; step < steps; ++step) {
                solver.template step<LMHFE>();
                solver.template chainA<LMHFE>(gWrtP);

                for (std::size_t cell = 0; cell < cells; ++cell) ret[step] += solver.scalarWrtA[cell];
            }

            return ret;
        }
    );
    m.def(
        "statFindiff",
        [] (std::size_t steps, Real da = 0.01) -> torch::Tensor {
            Solver baseSolver;

            initSolver(baseSolver, 10, 5);
            baseSolver.cache();

            const auto cells = baseSolver.getDomain().getMesh().template getEntitiesCount<Solver::dimCell>();

            std::vector<Solver> solvers(cells);
            for (std::size_t cell = 0; cell < cells; ++cell) {
                auto& s = solvers.at(cell);
                initSolver(s, 10, 5);
                s.a[cell] += da;
                s.cache();
            }

            const auto options = torch::TensorOptions().dtype(tdtReal<Real>);
            auto ret = torch::zeros(steps, options);

            for (std::size_t step = 0; step < steps; ++step) {
                using namespace noa::test::mhfe::methods;
                baseSolver.template step<LMHFE>();
                for (auto&& s : solvers) s.template step<LMHFE>();

                for (const auto& s : solvers) {
                    for (std::size_t cell = 0; cell < cells; ++cell) ret[step] += s.p[cell] - baseSolver.p[cell];
                }
                ret[step] /= (cells * da);
            }

            return ret;
        }
    );
}
