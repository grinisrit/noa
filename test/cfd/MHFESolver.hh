#pragma once

// Standard library
#include <string>

// Local headers
#include "MHFE.hh"

namespace noa::MHFE {

/// MHFE solver class
/// Stores the domain as well as provides the means to go over
/// MHFE/LMHFE solution manually via its static functions
template <typename DeltaFunc, typename LumpingFunc, typename RightFunc, __domain_targs__>
class Solver {
	/// Domain to operate on
	__DomainType__ domain;

	public:
	/// Should precise solution be written?
	bool writePrecise = false;
	/// Linear system solver name
	std::string solverName = "gmres";
	/// Linear system preconditioner name
	std::string preconditionerName = "diagonal";

	/// Cell dimension
	constexpr static auto dimCell = __DomainType__::getMeshDimension();
	/// Edge dimension
	constexpr static auto dimEdge = dimCell - 1;

	/// Cell layer getter
	///
	/// Will always return a Real layer, since all cell layers are of type `Real`
	template <Layer layer>
	auto& getCellLayer() {
		return this->domain.getLayers(dimCell)->template get<Real>(layer);
	} // <-- Solver::getCellLayer()
	template <Layer layer>
	const auto& getCellLayer() const { return const_cast<Solver*>(this)->template getEdgeLayer<layer>(); }
	
	/// Edge layer getter
	///
	/// Uses `if constexpr` to return appropriate layer type (`int`, `GlobalIndex` or `Real`)
	template <Layer layer>
	auto& getEdgeLayer() {
		if constexpr (layer == ROW_CAPACITIES)
			return this->domain.getLayers(dimEdge)->template get<GlobalIndex>(layer);
		else if constexpr ((layer == DIRICHLET_MASK) || (layer == NEUMANN_MASK))
			return this->domain.getLayers(dimEdge)->template get<int>(layer);
		else
			return this->domain.getLayers(dimEdge)->template get<Real>(layer);
	} // <-- Solver::getEdgeLayer()
	template <Layer layer>
	const auto& getEdgeLayer() const { return const_cast<Solver*>(this)->template getEdgeLayer<layer>(); }

	/// Mesh getter
	const auto& getMesh() const { return this->domain.getMesh(); }

	/// A constructor
	Solver(const __DomainType__& domain, const bool& writePrecise) :
								domain(domain),
								writePrecise(writePrecise)
	{
		// Prepare the domain copy
		prepareDomain(this->domain, this->writePrecise);
	} // <-- Solver::Solver()
	
	/// Initializer
	///
	/// Once all of the layers are initialized to some values, this function
	/// preforms some additional intial solver setup
	void init() {
		initDomain(this->domain);
	} // <-- Solver::init()
	
	/// Solver step
	///
	/// \param tau - time step
	void step(const Real& tau) {
		solverStep<DeltaFunc, LumpingFunc, RightFunc>(this->domain, tau, this->solverName, this->preconditionerName);
	} // <-- Solver::step()
}; // <-- class Solver

} // <-- namespace noa::MHFE
