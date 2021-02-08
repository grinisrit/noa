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
#include "ghmc/pms/dcs.hh"
#include "ghmc/utils.hh"

#include <torch/torch.h>

namespace ghmc::pms
{

    using namespace ghmc::utils;

    using Elements = std::vector<AtomicElement>;
    using ElementId = int;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using MaterialComposition =
        std::vector<std::tuple<ComponentFraction, ElementId>>;
    using Material = std::tuple<MaterialDensity, MaterialComposition>;
    using MaterialId = int;
    using Materials = std::vector<Material>;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using Shape = std::vector<int64_t>;
    using TableK = torch::Tensor;   // Kinetic energy tabulations
    using TableCSn = torch::Tensor; // CS normalisation tabulations

    inline const auto tfs_opt = torch::dtype(torch::kFloat32).layout(torch::kStrided);
    constexpr int NPR = 4;

    template <typename Physics, typename DCSKernels>
    class PhysicsModel
    {
        using DCSBremsstrahlung = typename std::tuple_element<0, DCSKernels>::type;
        using DCSPairProduction = typename std::tuple_element<1, DCSKernels>::type;
        using DCSPhotonuclear = typename std::tuple_element<2, DCSKernels>::type;
        using DCSIonisation = typename std::tuple_element<3, DCSKernels>::type;

    protected:
        inline Status check_mass(const mdf::MaterialsDEDXData &dedx_data)
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

        inline Status set_table_K(const mdf::MaterialsDEDXData &dedx_data)
        {
            auto data = dedx_data.begin();
            auto vals = std::get<mdf::DEDXTable>(data->second).T;
            auto n = vals.size();
            auto tensor = torch::from_blob(vals.data(), n, torch::kFloat32);
            table_K = static_cast<Physics *>(this)->scale_table_K(
                tensor.to(tfs_opt));
            data++;
            for (auto &it = data; it != dedx_data.end(); it++)
            {
                auto it_vals = std::get<mdf::DEDXTable>(it->second).T;
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

        inline Status perform_initial_checks(const mdf::Settings &mdf_settings,
                                             const mdf::MaterialsDEDXData &dedx_data)
        {
            if (!check_ZoA(mdf_settings, dedx_data))
                return false;
            if (!check_mass(dedx_data))
                return false;
            return true;
        }

        inline void set_elements(const mdf::Elements &mdf_elements)
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

        inline void set_materials(const mdf::Materials &mdf_materials)
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

        inline TableCSn compute_cs(ComputeCEL cel)
        {
            auto shape = Shape(3);
            shape[0] = elements.size();
            shape[1] = NPR;
            shape[2] = table_K.numel();
            auto table = torch::zeros(shape, tfs_opt);
            const auto &[br, pp, ph, io] = dcs_kernels;

#pragma omp parallel for
            for (int el = 0; el < shape[0]; el++)
            {
                eval_cs(br)(table[el][0], elements[el], mass, table_K, X_FRACTION, 180, cel);
                eval_cs(pp)(table[el][1], elements[el], mass, table_K, X_FRACTION, 180, cel);
                eval_cs(ph)(table[el][2], elements[el], mass, table_K, X_FRACTION, 180, cel);
                eval_cs(io)(table[el][3], elements[el], mass, table_K, X_FRACTION, 180, cel);
            }
            return table;
        }

        inline Status initialise_physics(
            const mdf::Settings &mdf_settings, const mdf::MaterialsDEDXData &dedx_data)
        {
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings));
            if (!set_table_K(dedx_data))
                return false;

            table_CSn = compute_cs(false);
            const auto table_cel = compute_cs(true);

            return true;
        }

    public:
        const DCSKernels dcs_kernels;
        const ParticleMass mass;
        const DecayLength ctau;

        PhysicsModel(DCSKernels dcs_kernels_, ParticleMass mass_, DecayLength ctau_)
            : dcs_kernels{dcs_kernels_}, mass{mass_}, ctau{ctau_} {}

        Elements elements;
        ElementIds element_id;

        Materials materials;
        MaterialIds material_id;

        TableK table_K;
        TableCSn table_CSn;

        inline Status load_physics_from(const mdf::Settings &mdf_settings,
                                        const mdf::MaterialsDEDXData &dedx_data)
        {
            if (!perform_initial_checks(mdf_settings, dedx_data))
                return false;
            if (!initialise_physics(mdf_settings, dedx_data))
                return false;
            return true;
        }
    };

    template <typename DCSKernels>
    class MuonPhysics : public PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>
    {

        friend class PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>;

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
        MuonPhysics(DCSKernels dcs_kernels_,
                    ParticleMass mass_ = MUON_MASS, DecayLength ctau_ = MUON_CTAU)
            : PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>(dcs_kernels_, mass_, ctau_) {}
    };

    template <typename DCSKernels>
    struct TauPhysics : MuonPhysics<DCSKernels>
    {
        TauPhysics(DCSKernels dcs_kernels_,
                   ParticleMass mass_ = TAU_MASS, DecayLength ctau_ = TAU_CTAU) : MuonPhysics<DCSKernels>(dcs_kernels_, mass_, ctau_) {}
    };

    template <typename PumasPhysics, typename DCSKernels>
    inline std::optional<PumasPhysics> load_pumas_physics_from(
        const mdf::ParticleName &particle_name, const mdf::MDFFilePath &mdf,
        const mdf::DEDXFolderPath &dedx, const DCSKernels &dcs_kernels)
    {
        if (!ghmc::utils::check_path_exists(mdf))
            return std::nullopt;
        if (!ghmc::utils::check_path_exists(dedx))
            return std::nullopt;

        auto mdf_settings = mdf::parse_settings(mdf::pumas, mdf);
        if (!mdf_settings.has_value())
            return std::nullopt;

        auto dedx_data = mdf::parse_materials(
            std::get<mdf::Materials>(mdf_settings.value()), dedx, particle_name);
        if (!dedx_data.has_value())
            return std::nullopt;

        auto pumas_physics = PumasPhysics(dcs_kernels);
        if (!pumas_physics.load_physics_from(*mdf_settings, *dedx_data))
            return std::nullopt;

        return pumas_physics;
    }

    template <typename DCSKernels>
    inline std::optional<MuonPhysics<DCSKernels>> load_muon_physics_from(
        const mdf::MDFFilePath &mdf, const mdf::DEDXFolderPath &dedx, const DCSKernels &dcs_kernels)
    {
        return load_pumas_physics_from<MuonPhysics<DCSKernels>, DCSKernels>(mdf::Muon, mdf, dedx, dcs_kernels);
    }

} // namespace ghmc::pms
