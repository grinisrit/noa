/*****************************************************************************
 *   Copyright (c) 2022, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Roland Grinis
 */

#pragma once

#include "noa/kernels.hh"
#include "noa/utils/common.hh"

#include <cstdio>

namespace noa::pms::pumas {

    // PUMAS type aliases
    using Medium        = pumas_medium;
    using Physics       = pumas_physics;
    using Particle      = pumas_particle;
    using Locals        = pumas_locals;
    using Step          = pumas_step;
    using Event         = pumas_event;

    using LocalsCb      = pumas_locals_cb;
    using LocalsCbFunc  = std::function<double(Medium*, class State*, Locals*)>;

    using MediumCb      = pumas_medium_cb;
    using MediumCbFunc  = std::function<Step(class Context*, class State*, Medium**, double*)>;

    // A wrapper for pumas_medium
    union MediumU {
            pumas_medium medium;
            struct Meta {
                std::size_t     medium_index;
                void*           model_ptr;
            } meta;
    };

    // A wrapper class for pumas_state
    class State {
            friend class Context; // State could only be constructed from a Context instance

            pumas_state         state{};
            class Context       *owner{nullptr};

            State(class Context* creator) : owner(creator) {}
            public:
            /*
            std::optional<std::size_t>          lastKnownTetrahedron{};
            std::optional<std::array<float, 3>> lastKnownDirection{};
            */
            // Access state fields via ->
            inline pumas_state * operator->() {
                    return &state;
            }
            inline const pumas_state * operator->() const {
                    return &state;
            }
            inline pumas_state & get() {
                    return state;
            }
            // Const version
            inline const pumas_state & get() const {
                    return state;
            }
    };

    // A wrapper class for pumas_context
    class Context {
        template <Particle default_particle> friend class PhysicsModel;

        pumas_context *context{nullptr};

        // Context is only create-able via PhysicsModel
        Context(Physics* physics) {
            switch (pumas_context_create(&this->context, physics, sizeof(this))) {
                case PUMAS_RETURN_SUCCESS:
                    *((Context**)this->context->user_data) = this;
                    this->context->medium = &Context::medium_callback;
                    return;
                case PUMAS_RETURN_MEMORY_ERROR:
                    std::cerr << "pumas_context_create: could not allocate memory!" << std::endl;
                    break;
                case PUMAS_RETURN_PHYSICS_ERROR:
                    std::cerr << "pumas_context_create: the physics is not initialized!" << std::endl;
                    break;
                default:
                    std::cerr << "pumas_context_create: unexpected error!" << std::endl;
                    break;
            }
            throw std::runtime_error("pumas_context_create failure!");
        }

        static Step medium_callback(
                pumas_context* context,
                pumas_state* state,
                Medium** medium_ptr,
                double* step_ptr) {
            auto* self = *((Context**)context->user_data);
            return self->medium(
                        self,
                        (State*)state, // We expect state to be wrapped in State
                        medium_ptr,
                        step_ptr);
        }

        public:
        MediumCbFunc medium{nullptr};

        ~Context() {
            this->destroy();
        }

        Context(Context &&other) noexcept : medium(std::move(other.medium)) {
            this->context = other.context;
            *((Context**)this->context->user_data) = this;
            other.context = nullptr;
        }
        Context & operator=(Context &&other) noexcept {
            medium = std::move(other.medium);
            this->context = other.context;
            *((Context**)this->context->user_data) = this;
            other.context = nullptr;
            return *this;
        }

        Context(const Context &other)               = delete;
        Context & operator=(const Context &other)   = delete;

        // Velocity sign (depends on mode direction FORWARD/BACKWARD)
        inline int sgn() { return (this->context->mode.direction == PUMAS_MODE_FORWARD) ? 1 : -1; }

        inline void destroy() {
            if (this->context != nullptr) pumas_context_destroy(&this->context);
        }

        inline Event do_transport(State &state, Medium *medium[2]) {
            Event ret;
            pumas_context_transport(this->context, &state.get(), &ret, medium);
            return ret;
        }

        inline State create_state() {
            return State(this);
        }

        inline pumas_context * operator->() { return this->context; }
        inline const pumas_context * operator->() const { return this->context; }

        inline auto rnd() { return this->context->random(this->context); }
    };
    using ContextOpt = std::optional<Context>;

    template<Particle default_particle = PUMAS_PARTICLE_MUON>
    class PhysicsModel {

        using PhysicsModelOpt = std::optional<PhysicsModel>;
        using MDFPath = utils::Path;
        using DEDXPath = utils::Path;
        using BinaryPath = utils::Path;

        Particle particle{default_particle};
        Physics *physics{nullptr};

        std::vector<MediumU> media{};
        std::vector<LocalsCbFunc> media_locals;

        explicit PhysicsModel(Particle particle_) : particle{particle_} {}

        inline utils::Status create_physics(
                const MDFPath &mdf_path,
                const DEDXPath &dedx_path) {
            if (utils::check_path_exists(mdf_path) && utils::check_path_exists(dedx_path)) {
                const auto status =
                        pumas_physics_create(&physics, particle, mdf_path.c_str(), dedx_path.c_str(), nullptr);
                return status == PUMAS_RETURN_SUCCESS;
            }
            return false;
        }

        inline utils::Status load_physics(const BinaryPath &binary_path) {
            if (utils::check_path_exists(binary_path)) {
                auto handle = fopen(binary_path.c_str(), "rb");
                if (handle == nullptr) {
                        perror(binary_path.string().c_str());
                        return false;
                }
                const auto status =
                        pumas_physics_load(&physics, handle);
                fclose(handle);
                return status == PUMAS_RETURN_SUCCESS;
            }
            return false;
        }

        static double locals_callback(Medium* medium, pumas_state* state, Locals* locals) {
                const auto* meta = (MediumU::Meta*)(medium + 1);
                const auto& idx = meta->medium_index;
                const auto* model = (PhysicsModel*)(meta->model_ptr);

                return model->media_locals.at(idx / 2)(medium, (State*)state, locals);
        }

    public:
        PhysicsModel() = default;

        PhysicsModel(
                const PhysicsModel &other) = delete;

        auto &operator=(const PhysicsModel &other) = delete;

        PhysicsModel(PhysicsModel &&other)
        noexcept
                : particle{other.particle}, physics{other.physics} {
            other.physics = nullptr;
        }

        auto &operator=(PhysicsModel &&other) noexcept {
            particle = other.particle;
            physics = other.physics;
            other.physics = nullptr;
            return *this;
        }

        ~PhysicsModel() {
            pumas_physics_destroy(&physics);
            physics = nullptr;
        }

        inline std::optional<int> get_material_index(const std::string& mat_name) const {
            int retval;
            switch (pumas_physics_material_index(this->physics, mat_name.c_str(), &retval)) {
                case PUMAS_RETURN_SUCCESS:
                    return retval;
                case PUMAS_RETURN_PHYSICS_ERROR:
                    std::cerr << __FUNCTION__ << ": The physics is not initialized!" << std::endl;
                    return std::nullopt;
                case PUMAS_RETURN_UNKNOWN_MATERIAL:
                    std::cerr << __FUNCTION__ << ": The material is not defined!" << std::endl;
                    return std::nullopt;
                default:
                    std::cerr << __FUNCTION__ << ": Unexpected error!" << std::endl;
                    return std::nullopt;
            }
        }

        inline ContextOpt create_context() {
            try {
                return ContextOpt{Context(this->physics)};
            } catch (const std::runtime_error& e) {
                return std::nullopt;
            }
        }

        inline std::optional<std::size_t> add_medium(const std::string& mat_name, const LocalsCbFunc& locals_func) {
            const auto mat_idx_opt = this->get_material_index(mat_name);
            if (!mat_idx_opt.has_value()) return {};
            const auto& mat_idx = mat_idx_opt.value();

            pumas_medium medium{ mat_idx, locals_callback };
            MediumU::Meta meta{ this->media.size(), this };
            this->media.push_back(MediumU{ .medium = medium });
            this->media.push_back(MediumU{ .meta = meta });
            this->media_locals.push_back(locals_func);

            return this->media.size() - 2;
        }

        inline Medium * get_medium(const int& mat_index) {
                for (std::size_t i = 0; i < this->media.size(); i += 2)
                        if (this->media.at(i).medium.material == mat_index) return &this->media.at(i).medium;
                return nullptr;
        }
        inline const Medium * get_medium(const int& mat_index) const {
                for (std::size_t i = 0; i < this->media.size(); i += 2)
                        if (this->media.at(i).medium.material == mat_index) return &this->media.at(i).medium;
                return nullptr;
        }

        inline Medium * get_medium(const std::string& mat_name) {
                const auto mat_index_opt = this->get_material_index(mat_name);
                if (!mat_index_opt.has_value()) return nullptr;
                return this->get_medium(mat_index_opt.value());
        }
        inline const Medium * get_medium(const std::string& mat_name) const {
                const auto mat_index_opt = this->get_material_index(mat_name);
                if (!mat_index_opt.has_value()) return nullptr;
                return this->get_medium(mat_index_opt.value());
        }

        inline void clear_media() {
            this->media.clear();
        }

        inline utils::Status save_binary(const BinaryPath &binary_path) const {

            auto handle = fopen(binary_path.c_str(), "wb");
            if (handle != nullptr) {
                pumas_physics_dump(physics, handle);
                fclose(handle);
            } else {
                std::cerr << "Failed to open " << binary_path << "\n";
                return false;
            }
            return true;
        }

        inline static PhysicsModelOpt load_from_mdf(
                const MDFPath &mdf_path,
                const DEDXPath &dedx_path) {
            auto model = PhysicsModel{};
            const auto status = model.create_physics(mdf_path, dedx_path);
            return status ? PhysicsModelOpt{std::move(model)} : PhysicsModelOpt{};
        }

        inline static PhysicsModelOpt load_from_binary(const BinaryPath &binary_path) {
            auto model = PhysicsModel{};
            const auto status = model.load_physics(binary_path);
            return status ? PhysicsModelOpt{std::move(model)} : PhysicsModelOpt{};
        }

    };

    using MuonModel     = PhysicsModel<PUMAS_PARTICLE_MUON>;
    using TauModel      = PhysicsModel<PUMAS_PARTICLE_TAU>;


}
