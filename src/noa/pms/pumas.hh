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

    template<typename ParticleModel>
    class PhysicsModel {

        using PhysicsModelOpt = std::optional<PhysicsModel>;
        using MDFPath = utils::Path;
        using DEDXPath = utils::Path;
        using DumpPath = utils::Path;

        friend ParticleModel;

        pumas_particle particle{PUMAS_PARTICLE_MUON};
        pumas_physics *physics{nullptr};

        explicit PhysicsModel(pumas_particle particle_) : particle{particle_} {}

        inline utils::Status create_physics(
                const MDFPath &mdf_path,
                const DEDXPath &dedx_path
        ) {
            const auto status =
                    pumas_physics_create(&physics, particle, mdf_path.c_str(), dedx_path.c_str(), nullptr);
            return status == PUMAS_RETURN_SUCCESS;
        }

    public:

        PhysicsModel(const PhysicsModel &other) = delete;

        auto &operator=(const PhysicsModel &other) = delete;

        PhysicsModel(PhysicsModel &&other) noexcept
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

        inline utils::Status save_binary(const DumpPath &dump_path) const {

            auto handle = fopen(dump_path.c_str(), "wb");
            if (handle != nullptr) {
                pumas_physics_dump(physics, handle);
                fclose(handle);
            } else {
                std::cerr << "Failed to open " << dump_path << "\n";
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