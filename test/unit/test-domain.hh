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

	auto& l1 = cellLayers.template add<float>(0, 3.1415f);
	l1.alias = "Float Layer";
	l1.exportHint = true;
	auto& l2 = cellLayers.template add<double>(1, 2.7182);
	l2.alias = "Double Layer";
	l2.exportHint = true;
	auto& l3 = cellLayers.template add<int>(2, 42);
	l3.alias = "Integer Layer";
	l3.exportHint = true;
	auto& l4 = cellLayers.template add<int>(3, 142);
	l4.alias = "Unsaved layer";

	constexpr auto test_domain_file = "domain-test-save-load.vtu";
	domain.write(test_domain_file);

	DomainType domain2;
	domain2.loadFrom(test_domain_file, { { "Float Layer", 0 }, { "Double Layer", 1 }, { "Integer Layer", 2 } });

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

	constexpr auto test_domain_file_2 = "domain-test-save-load-2.vtu";
	domain2.write(test_domain_file_2);

	ASSERT_EQ(utils::compare_files(test_domain_file, test_domain_file_2), true);
}
