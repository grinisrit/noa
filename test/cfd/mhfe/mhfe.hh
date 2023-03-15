/**
 * \file mhfe.hh
 * \brief Main file for MHFEM solver implementation
 *
 * Implemented by: Gregory Dushkin
 */

#pragma once

// Standard library
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/LinearSolverTypeResolver.h>

// NOA headers
#include <noa/utils/common.hh>
#include <noa/utils/domain/domain.hh>

// Local headers
#include "../utils/common.hh"
#include "except.hh"
#include "geometry.hh"
#include "layers.hh"
#include "methods.hh"

// getEntityMeasure should be included after Mesh.h
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityMeasure.h>

/// \brief A namespace containing MHFEM implementation
namespace noa::test::mhfe {

/// \brief Solver features
///
/// Describes various options that could be on or off for the solver
/// Passed to a \Solver as a type template parameter since features might
/// not be implemented for certain domain/method configurations
///
/// TODO: When and if moving to C++20, make bools bit fields with size of 1 bit?
struct Features {
	/// Is precise solution layer allocated by this solver?
	bool precise	= false;
	/// Is this solver used to calculate some scalar function
	/// derivative by a via adjoints?
	bool chainA	= false;

	/// Placeholder enum type to pass \ref Features to \ref Solver
	///
	/// Is only required since non-type template parameters are not
	/// available prior to C++20 and we don't want to just pass a
	/// regular `int` parameter to solver.
	enum class Enum : unsigned int {};

	/// \brief Construct a features object from Features::Enum enum
	///
	/// Is not a constructor because declaring a constructor removes
	/// list initialization from this struct
	constexpr static Features from(Enum initializer) noexcept {
		return Features{
			bool(static_cast<unsigned int>(initializer) & 1),
			bool(static_cast<unsigned int>(initializer) & (1 << 2))
		};
	}

	constexpr operator Enum() const noexcept {
		return static_cast<Enum>(
			(unsigned int)(this->precise) << 1 |
			(unsigned int)(this->chainA) << 2
		);
	}
};

/// System is a friend of \ref Solver and has access to
/// all of its members. Its only purpose is to, given a method,
/// calculate MHFEM system matrices and terms
///
/// Every system specialization should include matrixTerm() and rhsTerm() functions
template <Features::Enum solverFeatures, __domain_targs__>
struct System {}; // <-- struct System

/// \brief A MHFEM solver class
///
/// \tparam features_ - solver features. Its type being enum is a placeholder until
/// we move to C++20 FIXME
///
/// Other template parameters repeat those of \ref Domain
template <
	Features::Enum features_,
	typename CellTopology,
#ifdef HAVE_OPENMP
    typename Device = TNL::Devices::Sequential,
#else
	typename Device = TNL::Devices::Host,
#endif
	typename Real = float,
	typename GlobalIndex = long int,
	typename LocalIndex = short int
> class Solver {
public:
	// Member types and type aliases
	/// Domain type
	using DomainType = __DomainType__;
	/// System matrix type
	using SparseMatrixType	= TNL::Matrices::SparseMatrix<Real, Device, GlobalIndex>;
	using DenseMatrixType	= TNL::Matrices::DenseMatrix<Real, Device, GlobalIndex>;
private:
	using SystemType = System<features_, CellTopology, Device, Real, GlobalIndex, LocalIndex>;
	friend SystemType;

	static constexpr auto features = Features::from(features_); // TODO: Remove this with C++20

	/// TNL Vector view types
	template <typename DataType>
	using VectorViewType		= typename DomainType::LayerManagerType::Vector<DataType>::ViewType;
	template <typename DataType>
	using ConstVectorViewType	= typename DomainType::LayerManagerType::Vector<DataType>::ViewType;

	/// Solution domain
	DomainType domain;
	/// Domain mesh type
	using MeshType = typename DomainType::MeshType;

public:
	/// Cell dimension
	static constexpr auto dimCell = DomainType::getMeshDimension();
	/// Edge dimension
	static constexpr auto dimEdge = dimCell - 1;

	// Layer views
	using Empty = noa::utils::test::Empty;
public:
	/// Solution view
	VectorViewType<Real> p;
private:
	/// Solution view (previous step)
	VectorViewType<Real> pPrev;
public:
	/// `a` coefficient view
	VectorViewType<Real> a;
	/// `c` coefficient view
	VectorViewType<Real> c;
	/// Precise solution view
	std::conditional_t<features.precise, VectorViewType<Real>, Empty> precise;
    /// Scalar function derivative with respect to a view
    std::conditional_t<features.chainA, VectorViewType<Real>, Empty> scalarWrtA;

private:
	using BInvEType = VectorViewType<Real>;
	using BInvVType	= std::vector<BInvEType>;
	using BInvVVType= std::vector<BInvVType>;
	/// B<sup>-1</sup> views
	BInvVVType bInv;
	/// Cell measure view
	VectorViewType<Real>		measure;
	/// `l` view
	VectorViewType<Real>		l;
public:
	/// Dirichlet mask view
	VectorViewType<int>		dirichletMask;
	/// Dirichlet condition view
	VectorViewType<Real>		dirichlet;
	/// Neumann mask view
	VectorViewType<int>		neumannMask;
	/// Neumann condition view
	VectorViewType<Real>		neumann;
private:
	/// System matrix row capacities view
	VectorViewType<GlobalIndex>	capacities;
	/// Right-hand-side vector view
	VectorViewType<Real>		rhs;
public:
	/// Solution over the edges
	VectorViewType<Real>		tp;

private:
	/// System matrix (TNL solvers require shared pointers)
	std::shared_ptr<SparseMatrixType>	M;

public:
        /// Getter of const view to system matrix
        ///
        /// Intended use: testing/debugging
        auto getM() const { return this->M->getConstView(); }

	/// Solution time
	Real t = 0;
	/// Solution time step
	Real tau = 0.005;

	/// TNL linear system solver name
	std::string solverName = "gmres";
	/// TNL preconditioner name
	std::string preconditionerName = "diagonal";

private:
	/// Get some uncached cell-wise coefficients
	///
	/// \return a tuple of (lambda, alpha_i, alpha, beta)
	auto getCellParameters(GlobalIndex cell, const LocalIndex cellEdges) {
		const Real lambda = this->c[cell] * this->measure[cell] / this->tau;
		const Real alpha_i = 1.0 / this->l[cell];
		const Real alpha = cellEdges * alpha_i;
		const Real beta = lambda + this->a[cell] * alpha;

		return std::make_tuple(lambda, alpha_i, alpha, beta);
	} // <-- void getCellParameters(cell, cellEdges)
public:
	/// \brief Constructor
	///
	/// \param features_ - used solver features
	///
	/// Populates domain with layers
	Solver() noexcept = default; // <-- Solver()
	
	/// Reset the solver
	void reset() {
		this->t = 0.0;
		this->M = nullptr;
		this->domain.clear();
	} // <-- void Solver::reset()

	/// (Re)create domain layers
	void updateLayers() {
		// Clear the existing layers
		this->domain.clearLayers();

		constexpr auto addLayer = [] (auto& manager, auto index, auto& view, auto initializer) -> auto& {
			using DataType = decltype(initializer);
			auto& ret = manager.template add<DataType>(index, initializer);
			view.bind(ret.template get<DataType>().getView());
			return ret;
		};

		// Create cell layers
		auto& cellLayers = this->domain.getLayers(dimCell);

		auto& layerP = addLayer(cellLayers, CellLayers::P, this->p, Real{});
		layerP.alias = "Computed solution";
		layerP.exportHint = true;

		addLayer(cellLayers, CellLayers::P_PREV,	this->pPrev,	Real{});
		addLayer(cellLayers, CellLayers::A,		this->a,	Real{});
		addLayer(cellLayers, CellLayers::C,		this->c,	Real{});
		addLayer(cellLayers, CellLayers::MEASURE,	this->measure,	Real{});
		addLayer(cellLayers, CellLayers::L,		this->l,	Real{});

		if constexpr (features.precise) {
			auto& layerPrecise = addLayer(cellLayers, CellLayers::PRECISE, this->precise, Real{});
			layerPrecise.alias = "Precise solution";
			layerPrecise.exportHint = true;
		}

        if constexpr (features.chainA) {
            auto& layerChain = addLayer(cellLayers, CellLayers::CHAIN, this->scalarWrtA, Real{});
            layerChain.alias = "Chain rule scalar function sensitivity";
            layerChain.exportHint = true;
        }

		// BINV needs more than one layer
		const auto& mesh = this->domain.getMesh();
		const auto cellEdgesMax = [&mesh] () {
			LocalIndex ret = 0;
			mesh.template forAll<dimCell>([&mesh, &ret] (GlobalIndex cell) {
					auto cellEdges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);
					if (cellEdges > ret) ret = cellEdges;
			});
			return ret;
		} ();

		this->bInv.clear();
		this->bInv = BInvVVType(cellEdgesMax, BInvVType(cellEdgesMax, BInvEType{}));
		for (LocalIndex i = 0; i < cellEdgesMax; ++i) for (LocalIndex j = 0; j < cellEdgesMax; ++j) {
			const auto idx = CellLayers::BINV + i * cellEdgesMax + j;
			addLayer(cellLayers, idx, this->bInv[i][j], Real{});
		}

		/// Create edge layers
		auto& edgeLayers = this->domain.getLayers(dimEdge);
		addLayer(edgeLayers, EdgeLayers::DIRICHLET,	this->dirichlet,	Real{});
		addLayer(edgeLayers, EdgeLayers::NEUMANN,	this->neumann,		Real{});
		addLayer(edgeLayers, EdgeLayers::DIRICHLET_MASK,this->dirichletMask,	int{});
		addLayer(edgeLayers, EdgeLayers::NEUMANN_MASK,	this->neumannMask,	int{});
		addLayer(edgeLayers, EdgeLayers::TP,		this->tp,		Real{});
		addLayer(edgeLayers, EdgeLayers::ROW_CAPACITIES,this->capacities,	GlobalIndex{});
		addLayer(edgeLayers, EdgeLayers::RHS_VECTOR,	this->rhs,		Real{});
	} // <-- void updateLayers()

	/// Autofills some of the domain layers with cached data
	void cache() {
		const auto& mesh = domain.getMesh();

		mesh.template forAll<dimEdge>([this, &mesh] (GlobalIndex edge) {
			// Fill matrix row capacities layer
			this->capacities[edge] = 1;

			if (this->dirichletMask[edge] != 0 && this->neumannMask[edge] == 0) return;

			// Ajacent edge cells
			const auto cells = mesh.template getSuperentitiesCount<dimEdge, dimCell>(edge);

			for (LocalIndex cell = 0; cell < cells; ++cell) {
				// Cell global index
				const auto gCellIdx = mesh.template getSuperentityIndex<dimEdge, dimCell>(edge, cell);
				this->capacities[edge] += mesh.template getSubentitiesCount<dimCell, dimEdge>(gCellIdx) - 1;
			}
		});

		mesh.template forAll<dimCell>([this, &mesh] (GlobalIndex cell) {
			// Fill cell measures
			this->measure[cell] = TNL::Meshes::getEntityMeasure(mesh, mesh.template getEntity<dimCell>(cell));

			// l
			this->l[cell] = geometry::l(this->domain, cell, this->measure[cell]);

			// B^-1
			const auto edges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);
			TNL::Matrices::DenseMatrix<Real, Device, LocalIndex> mBinv(edges, edges);
			geometry::Binv(this->domain, mBinv, cell, this->measure[cell], this->l[cell]);

			for (LocalIndex i = 0; i < edges; ++i) for (LocalIndex j = 0; j < edges; ++j) {
				this->bInv[i][j][cell] = mBinv.getElement(i, j);
			}
		});
	} // <-- void cache()

	/// Fills the system matrix and RHS according to current state and provided time step
	template <template <typename, typename> typename Method>
	void updateSystem() {
		const auto& mesh = domain.getMesh();

		// Get edges count
		const auto edges = mesh.template getEntitiesCount<dimEdge>();

		// Create matrix if needed
		if (this->M == nullptr) this->M = std::make_shared<SparseMatrixType>(edges, edges);
		this->M->setRowCapacities(this->capacities);
		// Zero all elements
		this->M->forAllElements([] (auto row, auto loc, auto col, auto& v) { v = 0; });

		this->rhs.forAllElements([] (GlobalIndex i, Real& v) { v = 0; });

		// Get the matrix view
		mesh.template forAll<dimEdge>([this, &mesh] (GlobalIndex edge) {
			if (this->dirichletMask[edge] != 0) {
				this->M->addElement(edge, edge, 1, 1);
				this->rhs[edge] += this->dirichlet[edge];
			}

			if (this->dirichletMask[edge] != 0 && this->neumannMask[edge] == 0) return;

			const auto edgeCells = mesh.template getSuperentitiesCount<dimEdge, dimCell>(edge);

			for (LocalIndex lCell = 0; lCell < edgeCells; ++lCell) {
				const auto cell = mesh.template getSuperentityIndex<dimEdge, dimCell>(edge, lCell);
				// Get edges per cell
				const auto cellEdges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);

				SystemType::template matrixTerm<Method>(*this, cell, edge, cellEdges);
				SystemType::template rhsTerm<Method>(*this, cell, edge, cellEdges, this->neumannMask[edge] * this->neumann[edge]);
			}
		});
	} // <-- void updateSystem()

	/// Performs a single simulation step
	///
	/// \tparam Method - simulation method (see \ref noa::test::mhfe::methods)
	///
	/// \throws exceptions::invalid_setup if \ref isValid() returns `false`
	/// \throws exceptions::empty_setup if \ref domain is empty
	template <template <typename, typename> typename Method>
	void step() {
		if (this->domain.isClean())	throw exceptions::empty_setup{};
		if (!this->isValid())		throw exceptions::invalid_setup{};

		this->pPrev = this->p;
		this->template updateSystem<Method>();

		// Solve the system
		auto preconditioner = TNL::Solvers::getPreconditioner<SparseMatrixType>(this->preconditionerName);
		auto solver = TNL::Solvers::getLinearSolver<SparseMatrixType>(this->solverName);

		preconditioner->update(this->M);
		solver->setMatrix(this->M);
		//solver->setPreconditioner(preconditioner);

		solver->solve(this->rhs, this->tp);

		// Update cell-wise solution from edge-wise
		const auto& mesh = this->domain.getMesh();
		this->p.forAllElements([this, &mesh] (GlobalIndex cell, Real& pCell) {
			const auto cellEdges = this->domain.getMesh().template getSubentitiesCount<dimCell, dimEdge>(cell);
			const auto [ lambda, alpha_i, alpha, beta ] = this->getCellParameters(cell, cellEdges);

			pCell = this->pPrev[cell] * lambda / beta;

			for (LocalIndex lei = 0; lei < cellEdges; ++lei) {
				const auto gEdge = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lei);
				pCell += this->a[cell] * this->tp[gEdge] / beta / this->l[cell];
			}
		});

		// Advance the time
		this->t += this->tau;
	} // <-- void step()

private:
    // These are the private members responsible for what is happening inside of
    // `chainA`. This entire class needs to be rewritten so I don't feel TOO bad
    // bound declaring them here
    using RealVectorType = TNL::Containers::Vector<Real, Device, GlobalIndex>;

    std::conditional_t<features.chainA, RealVectorType, Empty>  g_wrt_p;
    std::conditional_t<features.chainA, DenseMatrixType, Empty> tp_wrt_a;
    std::conditional_t<features.chainA, DenseMatrixType, Empty> p_wrt_a;
    std::conditional_t<features.chainA, RealVectorType, Empty>  b_wrt_tp;
    std::conditional_t<features.chainA, std::vector<SparseMatrixType>, Empty> M_wrt_a;
    std::conditional_t<features.chainA, std::vector<RealVectorType>, Empty>   rhs_1;
    std::conditional_t<features.chainA, RealVectorType, Empty>  tp_wrt_ai;

public:
    /// \brief Perform an initial setup for chain rule calculation
    template <template <typename, typename> class Method, typename FuncWrtP>
    void chainAInit(FuncWrtP funcWrtP) {
		const auto& mesh = this->getDomain().getMesh();

		const auto numCells = mesh.template getEntitiesCount<dimCell>();
		const auto numEdges = mesh.template getEntitiesCount<dimEdge>();

        g_wrt_p = RealVectorType(numCells);
        tp_wrt_a = DenseMatrixType(numEdges, numCells);
        p_wrt_a = DenseMatrixType(numCells, numCells);

        tp_wrt_a.forAllElements([] (auto, auto, auto, auto& v) { v = 0; });
        p_wrt_a.forAllElements([] (auto, auto, auto, auto& v) { v = 0; });

        b_wrt_tp = RealVectorType(numEdges); // A diagonal matrix
        
        M_wrt_a = std::vector<SparseMatrixType>(numCells);
        for (auto&& M_wrt_ai : M_wrt_a) {
			M_wrt_ai = SparseMatrixType(numEdges, numEdges);
			M_wrt_ai.setRowCapacities(this->capacities);
        }

        rhs_1 = std::vector<RealVectorType>(numCells);
		for (auto& rhs_1_i : rhs_1) rhs_1_i = RealVectorType(numEdges);

        tp_wrt_ai = RealVectorType(numEdges);
    } // <-- void chainAInit()

	/// \biref Perfoms sensitivity calculation for a
	///
	/// Uses last step data
	template <template <typename, typename> class Method, typename FuncWrtP>
	Real chainA(FuncWrtP funcWrtP) {
		const auto& mesh = this->getDomain().getMesh();

		const auto numCells = mesh.template getEntitiesCount<dimCell>();
		const auto numEdges = mesh.template getEntitiesCount<dimEdge>();

		funcWrtP(this->p, numCells, g_wrt_p.getView());

		b_wrt_tp.forAllElements([&mesh, this] (auto edge, auto& v) {
			v = 0;
			if (this->dirichletMask[edge] > 0 && this->neumannMask[edge] == 0) return;

			const auto cells = mesh.template getSuperentitiesCount<dimEdge, dimCell>(edge);
			for (LocalIndex lCell = 0; lCell < cells; ++lCell) {
				const auto cell = mesh.template getSuperentityIndex<dimEdge, dimCell>(edge, lCell);
				const auto cellEdges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);
				v += this->measure[cell] * this->c[cell] / (cellEdges * this->tau);
			}
		});

		for (GlobalIndex cell = 0; cell < numCells; ++cell) {
			this->M->forAllElements([&mat = M_wrt_a[cell]] (auto row, auto, auto col, auto&) {
				mat.setElement(row, col, 0);
			});
			const auto cellEdges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);
			const auto [ lambda, alpha_i, alpha, beta ] = this->getCellParameters(cell, cellEdges);

			for (LocalIndex lEdge1 = 0; lEdge1 < cellEdges; ++lEdge1) {
				const auto edge1 = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lEdge1);
				if (this->dirichletMask[edge1] > 0) continue;

				for (LocalIndex lEdge2 = 0; lEdge2 < cellEdges; ++lEdge2) {
					const auto edge2 = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lEdge2);

					const auto delta = this->bInv[lEdge1][lEdge2][cell] - alpha_i * alpha_i / alpha;
					M_wrt_a[cell].addElement(edge1, edge2, delta, 1);
				}

			}
		}

		for (std::size_t i = 0; i < numCells; ++i) {
			// M_wrt_a * tp
			M_wrt_a[i].vectorProduct(this->tp, rhs_1[i]);

			rhs_1[i].forAllElements([cell=i, this] (auto edge, auto& v) {
				if (this->dirichletMask[edge] > 0) v = 0; // Dirichlet boundary condition
				// -M_wrt_a * tp + b_wrt_tp * tp_wrt_a
				else v = -v + b_wrt_tp[edge] * tp_wrt_a.getElement(edge, cell);
			});

			if (TNL::lpNorm(rhs_1[i], 2.0) <= std::numeric_limits<Real>::epsilon()) {
				rhs_1[i] = 0;
			}
		}

		// Solve the system
		auto preconditioner = TNL::Solvers::getPreconditioner<SparseMatrixType>(this->preconditionerName);
		auto solver = TNL::Solvers::getLinearSolver<SparseMatrixType>(this->solverName);

		preconditioner->update(this->M);
		solver->setMatrix(this->M);
		//solver->setPreconditioner(preconditioner);

		static constexpr auto npymat = [] (auto& stream, const auto& mat, std::size_t size) {
			stream << "[ ";
			for (std::size_t i = 0; i < size; ++i) {
				stream << "[ ";
				for (std::size_t j = 0; j < size; ++j) {
					stream << mat->getElement(i, j);
					if (j != size - 1) stream << ", ";
				}
				stream << " ]";
				if (i != size - 1) stream << ", ";
			}
			stream << " ]";
		};

		for (std::size_t cell = 0; cell < numCells; ++cell) {
			solver->solve(rhs_1[cell], tp_wrt_ai);
			for (std::size_t edge = 0; edge < numEdges; ++edge) {
				tp_wrt_a.setElement(edge, cell, tp_wrt_ai[edge]);
				if (std::isnan(tp_wrt_ai[edge])) {
					/*
					std::cout << "Solving for " << cell << std::endl;
					std::cout << "Syatem matrix: " << std::endl; //<< *this->M << std::endl;
					npymat(std::cout, this->M, numEdges);
					std::cout << std::endl;
					std::cout << "System right-hand-side vector: " << rhs_1[cell] << std::endl;
					*/

					std::cerr << "Full answer: " << tp_wrt_ai << std::endl;

					std::ofstream check("check.py");
					check << "import numpy as np" << std::endl;
					check << "import scipy as sp" << std::endl;
					check << "M = np.array(";
					npymat(check, this->M, numEdges);
					check << ")" << std::endl;
					check << "rhs = np.array(" << rhs_1[cell] << ")" << std::endl;
					check << "print('NumPy')" << std::endl;
					check << "print(np.linalg.solve(M, rhs))" << std::endl;
					check << "print('SciPy')" << std::endl;
					check << "print(sp.sparse.linalg.gmres(M, rhs))" << std::endl;
					throw std::runtime_error("Hit NaN");
				}
			}
		}

		p_wrt_a.forAllElements([&mesh, this] (auto cell, auto aCell, auto, auto& v) {
			const auto cellEdges = mesh.template getSubentitiesCount<dimCell, dimEdge>(cell);
			const auto [ lambda, alpha_i, alpha, beta ] = this->getCellParameters(cell, cellEdges);

			// V * p_wrt_a
			v *= lambda / beta;

			// + V_wrt_a * p
			if (cell == aCell) v -= this->pPrev[cell] * lambda * alpha / beta / beta;

			for (LocalIndex lei = 0; lei < cellEdges; ++lei) {
				const auto gEdge = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lei);

				// + U * tp_wrt_a
				v += this->a[cell] * tp_wrt_a.getElement(gEdge, aCell) * alpha_i / beta;

				// + U_wrt_a * tp
				if (cell == aCell) v += this->tp[gEdge] * alpha_i * lambda / beta / beta;
			}
		});

		p_wrt_a.vectorProduct(g_wrt_p, this->scalarWrtA);

		Real ret = 0;

		this->scalarWrtA.forAllElements([&ret] (auto, auto& v) { ret += v; });

		return ret;
	} // <-- void chainA()

	/// Checks whever the domain is set to correct conditions for solving
	[[nodiscard]] utils::Status isValid() noexcept {
		if (this->domain.isClean()) return false;

		bool ret = true;
		this->domain.getMesh().template forBoundary<dimEdge>([this, &ret] (GlobalIndex edge) {
			ret &= this->dirichletMask[edge] != 0 || this->neumannMask[edge] != 0;
		});
		return ret;
	} // <-- Status isValid()

	/// Domain getter
	[[nodiscard]] const auto& getDomain() const noexcept { return this->domain; }
	[[nodiscard]] auto& getDomain() noexcept { return this->domain; }
}; // <-- class Solver

// System specializations
template <Features::Enum features, __domain_targs_topospec__>
struct System<features, TNL::Meshes::Topologies::Triangle, Device, Real, GlobalIndex, LocalIndex> {
	using CellTopology = TNL::Meshes::Topologies::Triangle;
        using SolverType = Solver<features, CellTopology, Device, Real, GlobalIndex, LocalIndex>;
	template <template <typename, typename> typename Method>
	static inline
        void matrixTerm(SolverType& solver, GlobalIndex cell, GlobalIndex edge, GlobalIndex cellEdges) {
		constexpr auto dimCell = SolverType::dimCell;
		constexpr auto dimEdge = SolverType::dimEdge;

		const auto [ lambda, alpha_i, alpha, beta ] = solver.getCellParameters(cell, cellEdges);

		const auto& mesh = solver.domain.getMesh();

		// Local entity index for the edge
		LocalIndex local;
		for (LocalIndex lei = 0; lei < cellEdges; ++lei) {
			if (mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lei) == edge) {
				local = lei;
				break;
			}
		}

		for (LocalIndex lei = 0; lei < cellEdges; ++lei) {
			const auto gEdge = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, lei);

			const Real delta = Method<CellTopology, Real>::delta(
					solver.a[cell], solver.c[cell], solver.l[cell], solver.measure[cell],
					alpha, alpha_i, beta, lambda,
					solver.bInv[local][lei][cell], solver.p[cell], solver.tp[gEdge],
					solver.tau);

			solver.M->addElement(edge, gEdge, delta, 1);

			if (lei != local) continue;

			const Real lumping = Method<CellTopology, Real>::lumping(
					solver.a[cell], solver.c[cell], solver.l[cell], solver.measure[cell],
					alpha, alpha_i, beta, lambda,
					solver.bInv[local][lei][cell], solver.p[cell], solver.tp[gEdge],
					solver.tau);
			solver.M->addElement(edge, gEdge, lumping, 1);
		}
	} // <-- void matrixTerm()

	template <template <typename, typename> typename Method>
	static inline
	void rhsTerm(SolverType& solver, GlobalIndex cell, GlobalIndex edge, GlobalIndex cellEdges, Real right) {
		const auto [ lambda, alpha_i, alpha, beta ] = solver.getCellParameters(cell, cellEdges);

		solver.rhs[edge] += right + Method<CellTopology, Real>::rhs(
					solver.a[cell], solver.c[cell], solver.l[cell], solver.measure[cell],
					alpha, alpha_i, beta, lambda,
					0, solver.p[cell], solver.tp[edge], // Binv does not contribute to RHS
					solver.tau);
	} // <-- void rhsTerm()
}; // <-- struct System (Triangle)

} // <-- namespace noa::test::mhfe
