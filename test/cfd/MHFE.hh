#pragma once

// STL headers
#include <memory>
#include <string>
#include <tuple>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/LinearSolverTypeResolver.h>

// Local headers
#include "Domain.hh"
#include "Geometry.hh"
#include "Macros.hh"

// Still TNL but needs to be included after Mesh.h
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityMeasure.h>

namespace noa::MHFE {

enum Layer : std::size_t {
	// Cell layers
	P		= 0,
	P_PREV		= 1,
	A		= 2,
	C		= 3,
	L		= 4,
	// Edge layers
	DIRICHLET	= 0,
	NEUMANN		= 1,
	TP		= 2,
	DIRICHLET_MASK	= 3,
	NEUMANN_MASK	= 4,
	ROW_CAPACITIES	= 5,
	RIGHT		= 6,
}; // <-- enum Layer

// Prepare domain to work with the solver
template <DOMAIN_TARGS>
void prepareDomain(DOMAIN_TYPE& domain) {
	if (domain.isClean()) throw std::runtime_error("Cannot prepare an empty domain!");

	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	domain.clearLayers();

	// Set up the layers
	domain.getLayers(dimCell).template add<Real>(0);	// Index 0, P
	domain.getLayers(dimCell).template add<Real>(0);	// Index 1, P_PREV
	domain.getLayers(dimCell).template add<Real>(0);	// Index 2, A
	domain.getLayers(dimCell).template add<Real>(0);	// Index 3, C
	domain.getLayers(dimCell).template add<Real>(0);	// Index 4, l - cached values

	domain.getLayers(dimEdge).template add<Real>(0);	// Index 0, DIRICHLET
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 1, NEUMANN
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 2, TP
	domain.getLayers(dimEdge).template add<int>(0);		// Index 3, DIRICHLET_MASK
	domain.getLayers(dimEdge).template add<int>(0);		// Index 4, NEUMANN_MASK
	domain.getLayers(dimEdge).template add<GlobalIndex>(0);	// Index 5, ROW_CAPACITIES
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 6, RIGHT
}

template <DOMAIN_TARGS>
void checkDomain(DOMAIN_TYPE& domain) {
	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	const auto dMask = domain.getLayers(dimEdge).template get<int>(Layer::DIRICHLET_MASK).getConstView();
	const auto nMask = domain.getLayers(dimEdge).template get<int>(Layer::NEUMANN_MASK).getConstView();

	domain.getMesh().template forBoundary<dimEdge>([&] (GlobalIndex edge) {
		if ((dMask[edge] == 0) && (nMask[edge] == 0))
			throw std::runtime_error("No border conditions set on border edge " + std::to_string(edge));
	});
}

template <DOMAIN_TARGS>
void initDomain(DOMAIN_TYPE& domain) {
	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	const auto dMask = domain.getLayers(dimEdge).template get<int>(Layer::DIRICHLET_MASK).getConstView();
	const auto nMask = domain.getLayers(dimEdge).template get<int>(Layer::NEUMANN_MASK).getConstView();

	checkDomain(domain);

	const auto& mesh = domain.getMesh();

	// Fill capacities
	auto capacities = domain.getLayers(dimEdge).template get<GlobalIndex>(Layer::ROW_CAPACITIES).getView();
	domain.getMesh().template forAll<dimEdge>([&] (const GlobalIndex& edge) {
		capacities[edge] = 1;
		if ((dMask[edge] != 0) && (nMask(edge) == 0)) return;

		const auto cells = mesh.template getSuperentitiesCount<dimEdge, dimCell>(edge);

		for (int cell = 0; cell < cells; ++cell) {
			const auto gCellIdx = mesh.template getSuperentityIndex<dimEdge, dimCell>(edge, cell);
			capacities[edge] += mesh.template getSubentitiesCount<dimCell, dimEdge>(gCellIdx) - 1;
		}
	});

	// Fill l
	auto ls = domain.getLayers(dimCell).template get<Real>(Layer::L).getView();
	domain.getMesh().template forAll<dimCell>([&] (const GlobalIndex& cell) {
		const auto cellEntity = mesh.template getEntity<dimCell>(cell);
		const Real mes = TNL::Meshes::getEntityMeasure(mesh, cellEntity);
		ls[cell] = lGet(domain, cell, mes);
	});
}

template <DOMAIN_TARGS>
void solverStep(DOMAIN_TYPE& domain,
		const Real& tau,
		const std::string& solverName = "gmres",
		const std::string& preconditionerName = "diagonal") {
	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	const auto edges = domain.getMesh().template getEntitiesCount<dimEdge>();

	// Copy the previous solution
	auto& P = domain.getLayers(dimCell).template get<Real>(Layer::P);
	auto& PPrev = domain.getLayers(dimCell).template get<Real>(Layer::P_PREV);
	PPrev = P;

	// System matrix
	using Matrix = TNL::Matrices::SparseMatrix<Real, Device, GlobalIndex>;
	auto m = std::make_shared<Matrix>(edges, edges);
	m->setRowCapacities(domain.getLayers(dimEdge).template get<GlobalIndex>(Layer::ROW_CAPACITIES));
	m->forAllElements([] (GlobalIndex rowIdx, GlobalIndex localIdx, GlobalIndex& colIdx, Real& v) { v = 0; });
	auto mView = m->getView();

	// Needed layer views
	auto PView = P.getView();
	const auto PPrevView = PPrev.getConstView();
	auto rightView = domain.getLayers(dimEdge).template get<Real>(Layer::RIGHT).getView();
	auto TPView = domain.getLayers(dimEdge).template get<Real>(Layer::TP).getView();
	const auto dMask = domain.getLayers(dimEdge).template get<int>(Layer::DIRICHLET_MASK).getConstView();
	const auto nMask = domain.getLayers(dimEdge).template get<int>(Layer::NEUMANN_MASK).getConstView();
	const auto dView = domain.getLayers(dimEdge).template get<Real>(Layer::DIRICHLET).getConstView();
	const auto nView = domain.getLayers(dimEdge).template get<Real>(Layer::NEUMANN).getConstView();

	const auto aView = domain.getLayers(dimCell).template get<Real>(Layer::A).getConstView();
	const auto cView = domain.getLayers(dimCell).template get<Real>(Layer::C).getConstView();

	const auto lView = domain.getLayers(dimCell).template get<Real>(Layer::L).getConstView();

	// Reset the right-part vector
	rightView.forAllElements([] (GlobalIndex i, Real& v) { v = 0; });

	const auto Q_part = [&] (const GlobalIndex& cell, const GlobalIndex& edge, const Real& right = 0) {
		const auto a = aView[cell];
		const auto c = cView[cell];

		const auto cellEntity = domain.getMesh().template getEntity<dimCell>(cell);
		const Real mes = TNL::Meshes::getEntityMeasure(domain.getMesh(), cellEntity);

		const Real lambda = c * mes / tau;

		// TODO: Binv could be cached because they depend only on geometry
		const auto l = lView[cell];

		const auto cellEdges = domain.getMesh().template getSubentitiesCount<dimCell, dimEdge>(cell);
		TNL::Matrices::DenseMatrix<Real, Device, LocalIndex> mBinv(cellEdges, cellEdges);
		Binv(domain, mBinv, cell, mes, l);

		const auto alpha_i = 1 / l;
		const auto alpha = cellEdges * alpha_i;
		const auto beta = lambda + a * alpha;

		LocalIndex local;
		for (LocalIndex lei = cellEdges - 1; lei >= 0; --lei)
			if (domain.getMesh().template getSubentityIndex<dimCell, dimEdge>(cell, lei) == edge)
				local = lei;

		for (LocalIndex lei = cellEdges - 1; lei >= 0; --lei) {
			const Real delta = a * (mBinv.getElement(local, lei) - a / l / l / beta);
			const auto lumping = 0;
			const auto gEdge = domain.getMesh().template getSubentityIndex<dimCell, dimEdge>(cell, lei);
			mView.addElement(edge, gEdge, delta);
			if (lei == local) mView.addElement(edge, gEdge, lumping);
		}

		rightView[edge] += right + a * lambda / l / beta * PPrevView[cell];
	};

	domain.getMesh().template forAll<dimEdge>([&] (GlobalIndex edge) {
		if (dMask[edge] + nMask[edge] != 0) {
			if (dMask[edge] != 0) {
				mView.addElement(edge, edge, 1);
				rightView[edge] += dView[edge];
			}
			if (nMask[edge] != 0) {
				const auto cell = domain.getMesh().template getSuperentityIndex<dimEdge, dimCell>(edge, 0);
				Q_part(cell, edge, nView[edge]);
			}

			return;
		}

		const auto eCells = domain.getMesh().template getSuperentitiesCount<dimEdge, dimCell>(edge);
		for (GlobalIndex lCell = 0; lCell < eCells; ++lCell) {
			const auto cell = domain.getMesh().template getSuperentityIndex<dimEdge, dimCell>(edge, lCell);
			Q_part(cell, edge);
		}
	});

	auto stepSolver = TNL::Solvers::getLinearSolver<Matrix>(solverName);
	auto stepPrecond = TNL::Solvers::getPreconditioner<Matrix>(preconditionerName);

	stepPrecond->update(m);
	stepSolver->setMatrix(m);
	stepSolver->setPreconditioner(stepPrecond);

	stepSolver->solve(rightView, TPView);

	P.forAllElements([&] (GlobalIndex cell, Real& PCell) {
		const auto a = aView[cell];
		const auto c = cView[cell];

		const auto cellEntity = domain.getMesh().template getEntity<dimCell>(cell);
		const Real mes = TNL::Meshes::getEntityMeasure(domain.getMesh(), cellEntity);

		const auto cellEdges = domain.getMesh().template getSubentitiesCount<dimCell, dimEdge>(cell);

		const Real lambda = c * mes / tau;

		const auto l = lView[cell];
		const auto alpha_i = 1 / l;
		const auto alpha = cellEdges * alpha_i;
		const auto beta = lambda + a * alpha;

		PCell = PPrevView[cell] * lambda / beta;

		for (auto lei = cellEdges - 1; lei >= 0; --lei)
			PCell += a * TPView[domain.getMesh().template getSubentityIndex<dimCell, dimEdge>(cell, lei)] / beta / l;
	});
}

} // <-- namespace noa::MHFE
