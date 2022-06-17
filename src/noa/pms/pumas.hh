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

#include "noa/utils/common.hh"

#include <cstdio>

namespace noa::pms::pumas {

#include "noa/3rdparty/_pumas/pumas.h"

    // PUMAS type aliases
    using Physics       = pumas_physics;
    using Context       = pumas_context;
    using Particle      = pumas_particle;
    using Medium        = pumas_medium;
    using State         = pumas_state;
    using Locals        = pumas_locals;
    using LocalsCb      = pumas_locals_cb;
    using Step          = pumas_step;

    template<typename ParticleModel>
    class PhysicsModel {

        using PhysicsModelOpt = std::optional<PhysicsModel>;
        using MDFPath = utils::Path;
        using DEDXPath = utils::Path;
        using BinaryPath = utils::Path;

        friend ParticleModel;

        Particle particle{PUMAS_PARTICLE_MUON};
        Physics *physics{nullptr};

        std::vector<Medium> media{};

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

    public:
        Context *context{nullptr};
        Step (*medium_callback)(Context *context, State *state, Medium **medium_ptr, double *step_ptr);

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
            this->destroy_context();
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

        inline Context * create_context(const int &extra_memory = 0) {
            if (this->context != nullptr) {
                    std::cerr << __FUNCTION__ << ": Context is not free!" << std::endl;
                    return nullptr;
            }
            switch (pumas_context_create(&this->context, this->physics, extra_memory)) {
                case PUMAS_RETURN_SUCCESS:
                    return this->context;
                case PUMAS_RETURN_MEMORY_ERROR:
                    std::cerr << __FUNCTION__ << ": Could not allocate memory!" << std::endl;
                    return nullptr;
                case PUMAS_RETURN_PHYSICS_ERROR:
                    std::cerr << __FUNCTION__ << ": The physics is not initialized!" << std::endl;
                    return nullptr;
                default:
                    std::cerr << __FUNCTION__ << ": Unexpected error!" << std::endl;
                    return nullptr;
            }
        }

        inline void destroy_context() {
            pumas_context_destroy(&this->context);
        }

        inline std::optional<std::size_t> add_medium(const std::string& mat_name, LocalsCb* locals_func) {
            const auto mat_idx_opt = this->get_material_index(mat_name);
            if (!mat_idx_opt.has_value()) return {};
            const auto& mat_idx = mat_idx_opt.value();

            this->media.push_back({ mat_idx, locals_func });
            return this->media.size() - 1;
        }

        inline Medium * get_medium(const int& mat_index) {
                for (std::size_t i = 0; i < this->media.size(); ++i)
                        if (this->media.at(i).material == mat_index) return &this->media.at(i);
                return nullptr;
        }

        inline Medium * get_medium(const std::string& mat_name) {
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
            auto model = ParticleModel{};
            const auto status = model.create_physics(mdf_path, dedx_path);
            return status ? PhysicsModelOpt{std::move(model)} : PhysicsModelOpt{};
        }

        inline static PhysicsModelOpt load_from_binary(const BinaryPath &binary_path) {
            auto model = ParticleModel{};
            const auto status = model.load_physics(binary_path);
            return status ? PhysicsModelOpt{std::move(model)} : PhysicsModelOpt{};
        }

    };

    class MuonModel : public PhysicsModel<MuonModel> {
        friend class PhysicsModel<MuonModel>;

        MuonModel() : PhysicsModel<MuonModel>{PUMAS_PARTICLE_MUON} {}
    };

    class TauModel : public PhysicsModel<TauModel> {
        friend class PhysicsModel<TauModel>;

        TauModel() : PhysicsModel<TauModel>{PUMAS_PARTICLE_TAU} {}
    };


}
