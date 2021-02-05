/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "ghmc/pms/mdf.hh"
#include "ghmc/utils.hh"

#include <torch/torch.h>

namespace ghmc::pms
{

    using namespace ghmc::utils;

    using Elements = std::vector<AtomicElement>;
    using ElementId = int;
    using ElementIds = std::unordered_map<MDFElementName, ElementId>;
    using MaterialComposition =
        std::vector<std::tuple<ComponentFraction, ElementId>>;
    using Material = std::tuple<MaterialDensity, MaterialComposition>;
    using MaterialId = int;
    using Materials = std::vector<Material>;
    using MaterialIds = std::unordered_map<MDFMaterialName, MaterialId>;
    using TableK = torch::Tensor;

    template <typename Physics>
    class PhysicsModel
    {
    public:
        ParticleMass mass;
        DecayLength ctau;

        Elements elements;
        ElementIds element_id;

        Materials materials;
        MaterialIds material_id;

        // tabulated values for kinetic energy
        TableK table_K;

        inline Status load_physics_from(const MDFSettings &mdf_settings,
                                        const MaterialsDEDXData &dedx_data)
        {
            if (!perform_initial_checks(mdf_settings, dedx_data))
                return false;
            if (!initialise_physics(mdf_settings, dedx_data))
                return false;

            return true;
        }

    protected:

        inline Status check_mass(const MaterialsDEDXData &dedx_data)
        {
            for (const auto &[material, data] : dedx_data)
                if (static_cast<Physics *>(this)->scale_mass(
                        std::get<ParticleMass>(data)) != mass)
                {
                    std::cerr << "Inconsistent particle mass in "
                                 "dedx data for "
                              << material << std::endl;
                    return false;
                }
            return true;
        }

        inline Status set_table_K(const MaterialsDEDXData &dedx_data)
        {
            auto data = dedx_data.begin();
            auto vals = std::get<DEDXTable>(data->second).T;
            auto n = vals.size();
            auto tensor = torch::from_blob(vals.data(), n, torch::kFloat32);
            table_K = static_cast<Physics *>(this)->scale_table_K(
                tensor.to(torch::dtype(torch::kFloat32)
                              .layout(torch::kStrided)));
            data++;
            for (auto &it = data; it != dedx_data.end(); it++)
            {
                auto it_vals = std::get<DEDXTable>(it->second).T;
                auto it_ten = torch::from_blob(
                    it_vals.data(), n, torch::kFloat32);
                if (!torch::equal(tensor, it_ten))
                {
                    std::cerr
                        << "Inconsistent kinetic energy values for "
                        << it->first << std::endl;
                    return false;
                }
            }
            return true;
        }

        inline Status perform_initial_checks(const MDFSettings &mdf_settings,
                                             const MaterialsDEDXData &dedx_data)
        {
            if (!check_ZoA(mdf_settings, dedx_data))
                return false;
            if (!check_mass(dedx_data))
                return false;
            return true;
        }

        inline void set_elements(const MDFElements &mdf_elements)
        {
            int id = 0;
            elements.reserve(mdf_elements.size());
            for (auto [name, element] : mdf_elements)
            {
                element.I =
                    static_cast<Physics *>(this)->scale_excitation(element.I);
                elements.emplace_back(element);
                element_id.emplace(name, id);
                id++;
            }
        }

        inline void set_materials(const MDFMaterials &mdf_materials)
        {
            int id = 0;
            for (const auto &[name, material] : mdf_materials)
            {
                auto [_, density, components] = material;
                auto composition = MaterialComposition{};
                composition.reserve(components.size());
                for (const auto &[el, fraction] : components)
                {
                    auto el_id = element_id.at(el);
                    composition.emplace_back(fraction, el_id);
                }
                materials.emplace_back(
                    static_cast<Physics *>(this)->scale_density(density),
                    composition);
                material_id.emplace(name, id);
                id++;
            }
        }

        inline Status initialise_physics(
            const MDFSettings &mdf_settings, const MaterialsDEDXData &dedx_data)
        {
            set_elements(std::get<MDFElements>(mdf_settings));
            set_materials(std::get<MDFMaterials>(mdf_settings));
            if (!set_table_K(dedx_data))
                return false;
            return true;
        }
    };

    class MuonPhysics : public PhysicsModel<MuonPhysics>
    {

        friend class PhysicsModel<MuonPhysics>;

        inline ParticleMass scale_mass(const ParticleMass &mass)
        {
            return mass * 1E-3f; // from MeV to GeV
        }

        inline TableK scale_table_K(const TableK &table_K_)
        {
            return table_K_ * 1E-3f; // from MeV to GeV
        }

        inline MeanExcitation scale_excitation(const MeanExcitation &I)
        {
            return I * 1E-9f; // from eV to GeV
        }

        inline MaterialDensity scale_density(const MaterialDensity &density)
        {
            return density * 1E+3f; // from g/cm^3 to kg/m^3
        }

    public:
        MuonPhysics() : PhysicsModel{}
        {
            mass = MUON_MASS;
            ctau = MUON_CTAU;
        }
    };

    template <typename PumasPhysics>
    inline std::optional<PumasPhysics> load_pumas_physics_from(
        const ParticleName &particle_name, const MDFFilePath &mdf,
        const DEDXFolderPath &dedx)
    {
        if (!ghmc::utils::check_path_exists(mdf))
            return std::nullopt;
        if (!ghmc::utils::check_path_exists(dedx))
            return std::nullopt;

        auto mdf_settings = mdf_parse_settings("pumas", mdf);
        if (!mdf_settings.has_value())
            return std::nullopt;

        auto dedx_data = mdf_parse_materials(
            std::get<MDFMaterials>(mdf_settings.value()), dedx, particle_name);
        if (!dedx_data.has_value())
            return std::nullopt;

        auto pumas_physics = PumasPhysics{};
        if (!pumas_physics.load_physics_from(*mdf_settings, *dedx_data))
            return std::nullopt;

        return pumas_physics;
    }

    inline std::optional<MuonPhysics> load_muon_physics_from(
        const MDFFilePath &mdf, const DEDXFolderPath &dedx)
    {
        return load_pumas_physics_from<MuonPhysics>("Muon", mdf, dedx);
    }

} // namespace ghmc::pms
