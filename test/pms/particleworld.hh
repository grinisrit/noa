#pragma once

#include <noa/pms/pumas.hh>
#include <noa/pms/trace.hh>
#include <noa/utils/domain/domain.hh>

#include <limits>
#include <vector>
#include <string>

namespace noa::test::pms {

using namespace noa::pms;

// class ParticleWorld
// Creates a 'world' for PhysicsModel simulations
// The world could be inhabited by Domains, that contain
// meshes that alter environemnt physisc properties
template <pumas::Particle default_particle = pumas::PUMAS_PARTICLE_MUON, typename Real = float>
class ParticleWorld {
        // Type aliases are public for utility
        public:
        using CellTopology =    TNL::Meshes::Topologies::Tetrahedron;
        using DomainType =      utils::domain::Domain<CellTopology, TNL::Devices::Host, Real>;
        using MeshType =        typename DomainType::MeshType;
        using PointType =       typename MeshType::PointType;
        using ParticleModel =   pumas::PhysicsModel<default_particle>;
        using ModelOpt =        std::optional<ParticleModel>;
        using LocalsCbFunc =    pumas::LocalsCbFunc;
        using ContextOpt =      std::optional<pumas::Context>;
        using Tracer =          trace::Tracer<TNL::Devices::Host>;
        
        private:
        // World domains
        std::vector<DomainType> domains{};

        ModelOpt model{};
        ContextOpt context{};

        public:

        // World environment
        pumas::MediumCbFunc environment = nullptr;
        // Domain layer that contains medium data
        std::size_t medium_layer{};

        // Initialize PUMAS & physics simulation context
        void init(const utils::Path& dump_path, const utils::Path& materials_path = utils::Path()) {
                model = ParticleModel::load_from_binary(dump_path);
                if (!model.has_value()) {
                        // Model wasn't loaded from binary dump for some reason
                        constexpr auto mes = "Failed to load physics model from a binary dump";

                        if (materials_path.empty()) throw std::runtime_error(mes);
                        std::cerr << "Warning: " << mes << ". Trying MDF..." << std::endl;

                        const auto mdf_file = materials_path / "mdf" / "examples" / "standard.xml";
                        const auto dedx_dir = materials_path / "dedx";

                        model = ParticleModel::load_from_mdf(mdf_file, dedx_dir);

                        if (!model.has_value()) {
                                throw std::runtime_error("Failed to load physics model from MDF with "
                                                " materials path " + materials_path.string() + "!");
                        }

                        model.value().save_binary(dump_path);
                }

                context = model.value().create_context();
                if (!context.has_value())
                        throw std::runtime_error("Could not create PUMAS context");

                context->medium = [&model = this->model, &environment = this->environment, &domains = this->domains, &medium_layer = this->medium_layer] (pumas::Context* context_p, pumas::State* state_p, pumas::Medium** medium_p, double* step_p) -> pumas::Step {
                        if (environment == nullptr)
                                throw std::runtime_error("Environment medium is unset!");

                        if ((medium_p == nullptr) && (step_p == nullptr))
                                return pms::pumas::PUMAS_STEP_RAW;

                        const auto& state = *state_p;
                        PointType loc{};
                        for (std::size_t i = 0; i < 3; ++i) loc[i] = state->position[i];
                        PointType dir{};
                        for (std::size_t i = 0; i < 3; ++i) {
                                dir[i] = state->direction[i] * context_p->sgn();
                        }
                        auto speed = std::sqrt(TNL::dot(dir, dir));

                        for (const auto& domain : domains) {
                                const auto& mesh = domain.getMesh();
                                constexpr auto dim = domain.getMeshDimension(); // = 3
                                const auto cells = mesh.template getEntitiesCount<dim>();
                                /* PASEUDO-CODE
                                std::optional<std::size_t> t_index;
                                t_index = state.lastTetrahedronIndex;
                                if (t_index.has_value()) {
                                        if (!Trancer::check_in_tetrahedron(...)) t_index = std::nullopt;
                                }
                                // Search neighbours ...
                                if (!t_index.has_value()) t_index = Tracer::get_current_tetrahedron(&mesh, cells, loc);
                                state.lastTetrahedronIndex = t_index;

                                if (!t_index.has_value()) continue;
                                */
                                const auto t_index = Tracer::get_current_tetrahedron(&mesh, cells, loc);
                                if (!t_index.has_value()) continue;

                                const auto& cell_layers = domain.getLayers(dim);
                                const auto& medium_layer_data = cell_layers.template get<int>(medium_layer);

                                const int medium_index = medium_layer_data[t_index.value()];
                                if (medium_p != nullptr) *medium_p = model->get_medium(medium_index);

                                // Get the next boundary hit to get the step
                                const auto intersect = Tracer::get_first_border_in_tetrahedron(
                                                                        &mesh,
                                                                        t_index.value(),
                                                                        loc, dir,
                                                                        std::numeric_limits<float>::epsilon());

                                if (intersect.distance < 0)
                                        throw std::runtime_error("Intersection not found!");
                                /*
                                if (!intersect.is_intersection_with_triangle)
                                        std::cerr << "WARNING: Intersection not with triangle" << std::endl <<
                                                "Distance " << intersect.distance << "; location " << loc <<
                                                "; direction " << dir << std::endl;
                                */

                                // TODO: Maybe here we could check if the next tetrahedron contains
                                // the same material and extend the step, but we'll skip it for now
                                if (step_p != nullptr) {
                                        *step_p = intersect.distance;
                                        *step_p += std::numeric_limits<float>::epsilon();
                                }

                                return pumas::PUMAS_STEP_CHECK;
                        }

                        return environment(context_p, state_p, medium_p, step_p);
                };
        }

        std::optional<std::size_t> add_medium(const std::string& mat_name, const LocalsCbFunc& locals_func) {
                return model.value().add_medium(mat_name, locals_func);
        }

        DomainType& add_domain() {
                domains.emplace_back();
                return domains.back();
        }

        DomainType& get_domain(const std::size_t& idx) { return domains.at(idx); }
        const DomainType& get_domain(const std::size_t& idx) const { return domains.at(idx); }

        const ParticleModel&    get_model() const       { return model.value(); }
        pumas::Context&    get_context()           { return context.value(); }
}; // <-- class ParticleWorld

template <typename Real = float> using MuonWorld = ParticleWorld<pumas::PUMAS_PARTICLE_MUON, Real>;
template <typename Real = float> using TauWorld =  ParticleWorld<pumas::PUMAS_PARTICLE_TAU,  Real>;

} // <-- namespace noa::test::pms
