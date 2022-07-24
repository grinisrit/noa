#pragma once

#include <optional>

#include <noa/pms/pumas.hh>
#include <noa/pms/trace.hh>
#include <noa/utils/domain/domain.hh>

template <__domain_targs__>
inline std::optional<noa::pms::pumas::Step> domain_medium(
                double *step_ptr, const noa::utils::domain::Domain<__domain_targs__>& domain,
                const std::size_t& medium_index_layer,
                noa::pms::pumas::Context *context,
                noa::pms::pumas::State *state,
                noa::pms::pumas::Medium **medium_ptr
        ) {
        using Tracer = noa::pms::trace::Tracer<Device>;
        using Point = typename Tracer::Point;
        constexpr const auto& domainMesh = domain.getMesh();
        constexpr auto numTetrahedrons = domainMesh.template getEnititiesCount<domain.getMeshDimension()>();
        const auto p = Point((*state)->position[0], (*state)->position[1], (*state)->position[2]);
        const auto tetOpt = Tracer::get_current_tetrahedron(&domainMesh, numTetrahedrons, p);
        if (!tetOpt.has_value()) return {};
        return domain.getLayers(domain.getMeshDimension()).template get<int>(medium_index_layer)[tetOpt.value()];
}
