/**
 * \file geometry.hh
 * \brief Contains verious geometry dependant calculations
 *
 * Implemented by: Gregory Dushkin
 */

#pragma once

// Standard library
#include <vector>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityCenter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getOutwardNormalVector.h>

// NOA headers
#include <noa/utils/domain/domain.hh>

/// \brief Namespace for all geometry dependant functions
namespace noa::test::mhfe::geometry {

/// \brief Calculates geometric `l` coefficient.
///
/// Implementation is topology-dependant.
/// Accepts a \ref Domain reference instead of a Mesh reference to allow for a type deduction.
///
/// \param domain - a \ref Domain object reference
/// \param cell - cell index
/// \param measure - cell measure
template <__domain_targs__>
Real l(__DomainType__& domain, GlobalIndex cell, Real measure) {
	static_assert(
		std::is_same_v<CellTopology, nullptr_t>, // Always false, but will display the type of
							 // CellTopology in the compiler error log
		"l calculation is not implemented for selected topology"
	);
} // <-- l()

/// \brief Implementation for Triangular topology
template <__domain_targs_topospec__>
Real l(__DomainTypeTopoSpec__(TNL::Meshes::Topologies::Triangle)& domain, GlobalIndex cell, Real measure) {
	// Named constants for triangle topology
	constexpr auto dimCell = 2;
	constexpr auto dimEdge = 1;
	constexpr auto dimVert = 0;
	constexpr auto cellEdges = 3;

	auto& mesh = domain.getMesh();

	Real sqSum = 0;

	for (LocalIndex k = 0; k < cellEdges; ++k) { // Triangle topology 
		const auto edge = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, k);
		const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<dimEdge, dimVert>(edge, 0));
		const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<dimEdge, dimVert>(edge, 1));
		const auto r = p2 - p1;

		sqSum += (r, r);
	}

	return sqSum / 48.0 / measure;
} // <-- l() (Triangle)

/// \brief Calculates a B<sup>-1</sup> matrix
///
/// Implementation is topology-dependant.
/// 
/// \param domain - a \ref Domain object reference
/// \param matrix - matrix to store a result
/// \param cell - cell index
/// \param measure - cell measure
/// \param l - `l` coefficient, result of \ref l()
template <__domain_targs__, typename MatrixType>
void Binv(const __DomainType__& domain, MatrixType& matrix, GlobalIndex cell, Real measure, Real l) {
	static_assert(
		std::is_same_v<CellTopology, nullptr_t>, // Always false, but will display the type of
							 // CellTopology in the compiler error log
		"B^{-1} calculation is not implemented for selected topology"
	);
} // <-- Binv()

/// \brief Implementation of Triangular topology
template <__domain_targs_topospec__, typename MatrixType>
void Binv(const __DomainTypeTopoSpec__(TNL::Meshes::Topologies::Triangle)& domain,
		MatrixType& matrix, GlobalIndex cell, Real measure, Real l) {
	using CellTopology	= TNL::Meshes::Topologies::Triangle;
	using DomainType	= utils::domain::Domain<CellTopology, Device, Real, GlobalIndex, LocalIndex>;
	using PointType		= typename DomainType::MeshType::PointType;

	PointType r1(0, 0), r2(0, 0);

	// Named constants for triangle topology
	constexpr auto dimCell = 2;
	constexpr auto dimEdge = 1;
	constexpr auto dimVert = 0;
	constexpr auto cellEdges = 3;

	auto& mesh = domain.getMesh();

	const auto cellEntity = mesh.template getEntity<dimCell>(cell);
	const auto cellCenter = TNL::Meshes::getEntityCenter(mesh, cellEntity);

	std::vector<PointType> rv;

	for (LocalIndex k = 0; k < cellEdges; ++k) {
		const auto edge = mesh.template getSubentityIndex<dimCell, dimEdge>(cell, k);
		const auto p1 = mesh.getPoint(mesh.template getSubentityIndex<dimEdge, dimVert>(edge, 0));
		const auto p2 = mesh.getPoint(mesh.template getSubentityIndex<dimEdge, dimVert>(edge, 1));

		auto r = p2 - p1;

		const auto edgeEntity = mesh.template getEntity<dimEdge>(edge);
		auto n = TNL::Meshes::getOutwardNormalVector(mesh, edgeEntity, cellCenter);
		// Cross product Z coordinate
		const auto crossZ = n[0] * r[1] - n[1] * r[0];

		rv.push_back((1 - 2 * (crossZ < 0)) * r);
	}

	for (LocalIndex i = 0; i < cellEdges; ++i)
		for (LocalIndex j = 0; j < cellEdges; ++j)
			matrix.setElement(i, j, (rv[i], rv[j]) / measure + 1.0 / l / 3.0);
} // <-- Binv() (Triangle)

} // <-- namespace noa::test::mhfe::geometry
