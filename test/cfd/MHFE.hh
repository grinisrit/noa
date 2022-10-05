#pragma once

// STL headers
#include <functional>
#include <memory>
#include <string>
#include <tuple>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/LinearSolverTypeResolver.h>

// Local headers
#include <noa/utils/domain/domain.hh>
#include "Geometry.hh"

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
	PRECISE		= 5,
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
template <__domain_targs__>
void prepareDomain(__DomainType__& domain, const bool& allocatePrecise = false) {
	if (domain.isClean()) throw std::runtime_error("Cannot prepare an empty domain!");

	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	domain.clearLayers();

	// Set up the layers
	domain.getLayers(dimCell).template add<Real>(0);	// Index 0, P
	domain.getLayers(dimCell).getLayer(Layer::P).alias = "Computed Solution";
	domain.getLayers(dimCell).getLayer(Layer::P).exportHint = true;
	domain.getLayers(dimCell).template add<Real>(0);	// Index 1, P_PREV
	domain.getLayers(dimCell).template add<Real>(0);	// Index 2, A
	domain.getLayers(dimCell).template add<Real>(0);	// Index 3, C
	domain.getLayers(dimCell).template add<Real>(0);	// Index 4, l - cached values
	if (allocatePrecise) {
		domain.getLayers(dimCell).template add<Real>(0);// Index 5, PRECISE
		domain.getLayers(dimCell).getLayer(Layer::PRECISE).alias = "Precise Solution";
		domain.getLayers(dimCell).getLayer(Layer::PRECISE).exportHint = true;
	}

	domain.getLayers(dimEdge).template add<Real>(0);	// Index 0, DIRICHLET
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 1, NEUMANN
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 2, TP
	domain.getLayers(dimEdge).template add<int>(0);		// Index 3, DIRICHLET_MASK
	domain.getLayers(dimEdge).template add<int>(0);		// Index 4, NEUMANN_MASK
	domain.getLayers(dimEdge).template add<GlobalIndex>(0);	// Index 5, ROW_CAPACITIES
	domain.getLayers(dimEdge).template add<Real>(0);	// Index 6, RIGHT
}

template <__domain_targs__>
void checkDomain(__DomainType__& domain) {
	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	const auto dMask = domain.getLayers(dimEdge).template get<int>(Layer::DIRICHLET_MASK).getConstView();
	const auto nMask = domain.getLayers(dimEdge).template get<int>(Layer::NEUMANN_MASK).getConstView();

	domain.getMesh().template forBoundary<dimEdge>([&] (GlobalIndex edge) {
		if ((dMask[edge] == 0) && (nMask[edge] == 0))
			throw std::runtime_error("No border conditions set on border edge " + std::to_string(edge));
	});
}

template <__domain_targs__>
void initDomain(__DomainType__& domain) {
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
		if ((dMask[edge] != 0) && (nMask[edge] == 0)) return;

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

template <typename DeltaFunctor, typename LumpingFunctor, typename RightFunctor, __domain_targs__>
void solverStep(__DomainType__& domain,
		const Real& tau,
		const std::string& solverName = "gmres",
		const std::string& preconditionerName = "diagonal") {
	// Alias dimensions
	constexpr auto dimCell = domain.getMeshDimension();
	constexpr auto dimEdge = dimCell - 1;

	// Get edges count
	const auto edges = domain.getMesh().template getEntitiesCount<dimEdge>();

	// Copy the previous solution
	auto& P = domain.getLayers(dimCell).template get<Real>(Layer::P);
	auto& PPrev = domain.getLayers(dimCell).template get<Real>(Layer::P_PREV);
	PPrev = P;

	// System matrix - initialize and zero
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

	// Reset the right-hand vector
	rightView.forAllElements([] (GlobalIndex i, Real& v) { v = 0; });

	// Calculate system matrix and right-hand vector
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
			const Real delta	= DeltaFunctor::template get<CellTopology, Real>(alpha_i, alpha,
										lambda, beta,
										a, c, l,
										mes, tau,
										mBinv.getElement(local, lei));
			const Real lumping	= LumpingFunctor::template get<CellTopology, Real>(alpha_i, alpha,
										lambda, beta,
										a, c, l,
										mes, tau,
										mBinv.getElement(local, lei));
			const auto gEdge = domain.getMesh().template getSubentityIndex<dimCell, dimEdge>(cell, lei);
			mView.addElement(edge, gEdge, delta);
			if (lei == local) mView.addElement(edge, gEdge, lumping);
		}

		rightView[edge] += right + RightFunctor::template get<CellTopology, Real>(alpha_i, alpha,
										lambda, beta,
										a, c, l,
										mes, tau,
										PPrevView[cell], TPView[edge]);
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

template <typename SolFunc, __domain_targs__>
void writePrecise(__DomainType__& domain, SolFunc& solution, const Real& t) {
	// TODO, FIXME: this will only work for 2D
	constexpr auto dimCell = domain.getMeshDimension();

	auto preciseLayer = domain.getLayers(dimCell).template get<Real>(Layer::PRECISE).getView();
	const auto aView = domain.getLayers(dimCell).template get<Real>(Layer::A).getConstView();
	const auto cView = domain.getLayers(dimCell).template get<Real>(Layer::C).getConstView();
	const auto& mesh = domain.getMesh();

	mesh.template forAll<dimCell>([&t, &solution, &aView, &cView, &mesh, &preciseLayer] (GlobalIndex cell) {
			const auto cellEntity = mesh.template getEntity<dimCell>(cell);

			const auto verts = mesh.template getSubentitiesCount<dimCell, 0>(cell);
			typename __DomainType__::MeshType::PointType midpoint{0, 0};
			for (GlobalIndex vert = 0; vert < verts; ++vert)
				midpoint += mesh.getPoint(mesh.template getSubentityIndex<dimCell, 0>(cell, vert));
			preciseLayer[cell] = solution(midpoint[0] / verts, midpoint[1] / verts, t,
							aView[cell], cView[cell]);
	});
}

// Simulation steps take long enough that we can afford to call an std::function after each one
// Inlining won't make a big difference here
template <typename Real> using PostStepCb = std::function<void(const Real& t)>;
// Simulate up to a certain time
template <typename DeltaFunc, typename LumpingFunc, typename RightFunc, __domain_targs__>
void simulateTo(__DomainType__& domain, const Real& T, const Real& tau = .005, const PostStepCb<Real>& cb = nullptr) {
	Real t = 0;
	do {
		std::cout << "\r[" << std::setw(10) << std::left << t << "/" << std::right << T << "]";
		std::cout.flush();
		MHFE::solverStep<DeltaFunc, LumpingFunc, RightFunc>(domain, tau);
		t += tau;
		if (cb != nullptr) cb(t);
	} while (t < T);
	std::cout << " DONE" << std::endl;
}

template <typename DeltaFunc, typename LumpingFunc, typename RightFunc, typename TestFunc, __domain_targs__>
Real testCellSensitivityAt(__DomainType__ domain, // Layer will be altered, need a copy of a domain
			const Layer& dataLayer, // A cell layer that tester function will use
			const Layer& sensorLayer, // A cell layer that will be altered
			const Real& delta, // How much will the value be altered?
			const Real& T, const Real& tau = .005) {
	auto domain2 = domain; // Create another copy of the domain
	simulateTo<DeltaFunc, LumpingFunc, RightFunc>(domain, T, tau);
	constexpr auto cellDim = domain2.getMeshDimension();
	auto sensorView = domain2.getLayers(cellDim).template get<Real>(sensorLayer).getView();
	sensorView.forAllElements([&delta] (GlobalIndex i, Real& v) { v += delta; });
	simulateTo<DeltaFunc, LumpingFunc, RightFunc>(domain2, T, tau);

	// Now calculate tester over two domains and return the sensitivity
	return TestFunc::calc(domain2, dataLayer) - TestFunc::calc(domain, dataLayer);
}

} // <-- namespace noa::MHFE
