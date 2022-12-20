/**
 * \file mhfe.hh
 * \brief Main file for MHFEM solver implementation
 *
 * Implemented by: Gregory Dushkin
 */

#pragma once

// Standard library
#include <cassert>
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
#include "except.hh"
#include "geometry.hh"
#include "layers.hh"
#include "methods.hh"

// getEntityMeasure should be included after Mesh.h
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityMeasure.h>

/// \brief A namespace containing MHFEM implementation
namespace noa::test::mhfe {

/// System is a friend of \ref Solver and has access to
/// all of its members. Its only purpose is to, given a method,
/// calculate MHFEM system matrices and terms
///
/// Every system specialization should include matrixTerm() and rhsTerm() functions
template <__domain_targs__>
struct System {}; // <-- struct System

/// \brief A MHFEM solver class
///
/// Template parameters repeat those of \ref Domain
template <
	typename CellTopology,
	typename Device = TNL::Devices::Host,
	typename Real = float,
	typename GlobalIndex = long int,
	typename LocalIndex = short int
> class Solver {
public:
	// Member types and type aliases
	/// Domain type
	using DomainType = __DomainType__;
	/// System matrix type
	using MatrixType = TNL::Matrices::SparseMatrix<Real, Device, GlobalIndex>;
private:
	using SystemType = System<CellTopology, Device, Real, GlobalIndex, LocalIndex>;
	friend SystemType;

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
	VectorViewType<Real> precise;

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
	/// System matrix
	std::shared_ptr<MatrixType>	M;

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
	/// Is precise solution layer allocated by this solver?
	bool allocatePrecise;

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
	/// \param allocatePrecise_ - sets allocatePrecise member
	///
	/// Populates domain with layers
	Solver(bool allocatePrecise_ = false) noexcept : allocatePrecise(allocatePrecise_) {} // <-- Solver()

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

		if (this->allocatePrecise) {
			auto& layerPrecise = addLayer(cellLayers, CellLayers::PRECISE, this->precise, Real{});
			layerPrecise.alias = "Precise solution";
			layerPrecise.exportHint = true;
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
			TNL::Matrices::DenseMatrix<Real, TNL::Devices::Host, LocalIndex> mBinv(edges, edges);
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

		if (this->M == nullptr) this->M = std::make_shared<MatrixType>(edges, edges);

		this->M->setRowCapacities(this->capacities);

		// Zero all elements
		this->M->forAllElements([] (GlobalIndex row, GlobalIndex loc, GlobalIndex col, Real& v) { v = 0; });
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
		auto preconditioner = TNL::Solvers::getPreconditioner<MatrixType>(this->preconditionerName);
		auto solver = TNL::Solvers::getLinearSolver<MatrixType>(this->solverName);

		preconditioner->update(this->M);
		solver->setMatrix(this->M);
		solver->setPreconditioner(preconditioner);

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
template <__domain_targs_topospec__>
struct System<TNL::Meshes::Topologies::Triangle, Device, Real, GlobalIndex, LocalIndex> {
	using CellTopology = TNL::Meshes::Topologies::Triangle;
        using SolverType = Solver<CellTopology, Device, Real, GlobalIndex, LocalIndex>;
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
