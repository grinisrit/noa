#pragma once

// STL headers
// ...

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityCenter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getOutwardNormalVector.h>

// Local headers
#include <noa/utils/domain/domain.hh>

namespace noa::MHFE {

using noa::utils::domain::Domain;
using TNL::Meshes::Topologies::Triangle;

template <__domain_targs__>
Real lGet(const __DomainType__& domain,
		const GlobalIndex& cell,
		const Real& measure) {
	throw std::runtime_error("lGet is not implemented for selected topology!");
}

template <typename Device, typename Real, typename GlobalIndex, typename LocalIndex>
Real lGet(const Domain<Triangle, Device, Real, GlobalIndex, LocalIndex>& domain,
		const GlobalIndex& cell,
		const Real& measure) {
	constexpr auto dimCell = 2;
	constexpr auto dimEdge = 1;
	constexpr auto dimVert = 0;

	const auto cellEntity = domain.getMesh().template getEntity<dimCell>(cell);

	Real sqSum = 0;

	for (LocalIndex k = 0; k < 3; ++k) { // Triangle topology: 3 edges per cell
		const auto edge = domain.getMesh().template getSubentityIndex<dimCell, dimEdge>(cell, k);
		const auto p1 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<dimEdge, dimVert>(edge, 0));
		const auto p2 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<dimEdge, dimVert>(edge, 1));
		const auto r = p2 - p1;

		sqSum += (r, r);
	}

	return sqSum / 48.0 / measure;
}

template <__domain_targs__, typename MatrixType>
void Binv(const __DomainType__& domain,
		MatrixType& matrix,
		const GlobalIndex& cell,
		const Real& measure,
		const Real& l) {
	throw std::runtime_error("Binv is not implemented for selected topology");
}

template <typename Device, typename Real, typename GlobalIndex, typename LocalIndex, typename MatrixType>
void Binv(const Domain<Triangle, Device, Real, GlobalIndex, LocalIndex>& domain,
		MatrixType& matrix,
		const GlobalIndex& cell,
		const Real& measure,
		const Real& l) {
	using DomainType = Domain<Triangle, Device, Real, GlobalIndex, LocalIndex>;
	using PointType = typename DomainType::MeshType::PointType;
	PointType r1(0, 0), r2(0, 0);

	constexpr auto dimCell = 2;
	constexpr auto dimEdge = 1;
	constexpr auto dimVert = 0;

	const auto cellEntity = domain.getMesh().template getEntity<2>(cell);
	const auto cellCenter = TNL::Meshes::getEntityCenter(domain.getMesh(), cellEntity);

	std::vector<PointType> rv;

	for (LocalIndex k = 0; k < 3; ++k) {
		const auto edge = domain.getMesh().template getSubentityIndex<2, 1>(cell, k);
		const auto p1 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<1, 0>(edge, 0));
		const auto p2 = domain.getMesh().getPoint(domain.getMesh().template getSubentityIndex<1, 0>(edge, 1));

		auto r = p2 - p1;

		const auto edgeEntity = domain.getMesh().template getEntity<1>(edge);
		auto n = TNL::Meshes::getOutwardNormalVector(domain.getMesh(), edgeEntity, cellCenter);
		const auto crossZ = n[0] * r[1] - n[1] * r[0];

		rv.push_back((1 - 2 * (crossZ < 0)) * r);
	}

	for (LocalIndex i = 0; i < 3; ++i)
		for (LocalIndex j = 0; j < 3; ++j)
			matrix.setElement(i, j, (rv.at(i), rv.at(j)) / measure + 1.0 / l / 3.0);
}

} // <-- namespace noa::MHFE
