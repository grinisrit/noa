#pragma once

#include <noa/utils/domain/domain.hh>

#include <gtest/gtest.h>

TEST(DOMAIN, Create2DGridTriangle) {
	using namespace noa;

	using DomainType = utils::domain::Domain<TNL::Meshes::Topologies::Triangle>;

	DomainType domain;

	constexpr auto N = 10;
	utils::domain::generate2DGrid(domain, N, N, 1.0, 1.0);

	ASSERT_EQ(domain.getMesh().template getEntitiesCount<domain.getMeshDimension()>(), N * N * 2);
}

TEST(DOMAIN, SaveLoadDomain) {
	using namespace noa;

	using DomainType = utils::domain::Domain<TNL::Meshes::Topologies::Triangle>;

	DomainType domain;

	constexpr auto N = 10;
	utils::domain::generate2DGrid(domain, N, N, 1.0, 1.0);

	constexpr auto dimCell = domain.getMeshDimension();

	auto& cellLayers = domain.getLayers(dimCell);

	const auto l1 = cellLayers.template add<float>(3.1415f);
	cellLayers.getLayer(l1).alias = "Float Layer";
	cellLayers.getLayer(l1).exportHint = true;
	const auto l2 = cellLayers.template add<double>(2.7182);
	cellLayers.getLayer(l2).alias = "Double Layer";
	cellLayers.getLayer(l2).exportHint = true;
	const auto l3 = cellLayers.template add<int>(42);
	cellLayers.getLayer(l3).alias = "Integer Layer";
	cellLayers.getLayer(l3).exportHint = true;
	const auto l4 = cellLayers.template add<int>(142);
	cellLayers.getLayer(l4).alias = "Unsaved layer";

	constexpr auto test_domain_file = "domain-test-save-load.vtu";
	domain.write(test_domain_file);

	DomainType domain2;
	domain2.loadFrom(test_domain_file, { "Float Layer", "Double Layer", "Integer Layer" });

	ASSERT_EQ(domain.getMesh().template getEntitiesCount<dimCell>(), domain2.getMesh().template getEntitiesCount<dimCell>());

	std::size_t i = 0;
	ASSERT_EQ(cellLayers.getLayer(i).alias, domain2.getLayers(dimCell).getLayer(i).alias);
	ASSERT_EQ(cellLayers.template get<float>(i), domain2.getLayers(dimCell).template get<float>(i));
	i = 1;
	ASSERT_EQ(cellLayers.getLayer(i).alias, domain2.getLayers(dimCell).getLayer(i).alias);
	ASSERT_EQ(cellLayers.template get<double>(i), domain2.getLayers(dimCell).template get<double>(i));
	i = 2;
	ASSERT_EQ(cellLayers.getLayer(i).alias, domain2.getLayers(dimCell).getLayer(i).alias);
	ASSERT_EQ(cellLayers.template get<int>(i), domain2.getLayers(dimCell).template get<int>(i));

	domain2.write("domain-test-save-load-2.vtu");
}
