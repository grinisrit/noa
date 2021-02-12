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
#include "ghmc/pms/physics.hh"
#include "ghmc/utils.hh"

#include <torch/torch.h>

namespace ghmc::pms
{

    using namespace ghmc::utils;

    using Elements = std::vector<AtomicElement>;
    using ElementId = int;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ElementNames = std::vector<mdf::ElementName>;

    struct Material
    {
        torch::Tensor element_ids;
        torch::Tensor fractions;
    };
    using Materials = std::vector<Material>;
    using MaterialId = int;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using MaterialNames = std::vector<mdf::MaterialName>;

    using MaterialsDensity = torch::Tensor;
    using MaterialsZoA = torch::Tensor;
    using MaterialsI = torch::Tensor;
    using MaterialsDensityEffect = std::vector<MaterialDensityEffect>;

    using Shape = std::vector<int64_t>;
    using TableK = torch::Tensor;                // Kinetic energy tabulations
    using TableCSn = torch::Tensor;              // CS normalisation tabulations
    using TableCSf = std::vector<torch::Tensor>; // CS fractions by material

    inline const auto default_ops = torch::dtype(torch::kFloat64).layout(torch::kStrided);

    template <typename Physics, typename DCSKernels>
    class PhysicsModel
    {
        using DCSBremsstrahlung = typename std::tuple_element<0, DCSKernels>::type;
        using DCSPairProduction = typename std::tuple_element<1, DCSKernels>::type;
        using DCSPhotonuclear = typename std::tuple_element<2, DCSKernels>::type;
        using DCSIonisation = typename std::tuple_element<3, DCSKernels>::type;

        Elements elements;
        ElementIds element_id;
        ElementNames element_name;

        Materials materials;
        MaterialIds material_id;
        MaterialNames material_name;

        MaterialsDensity material_density;
        MaterialsZoA material_ZoA;
        MaterialsI material_I;
        MaterialsDensityEffect material_density_effect;

        TableK table_K;
        TableCSn table_CSn;
        TableCSf table_CSf;

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
            auto tensor = torch::from_blob(vals.data(), n, torch::kFloat64);
            table_K = static_cast<Physics *>(this)->scale_table_K(
                tensor.to(default_ops));
            data++;
            for (auto &it = data; it != dedx_data.end(); it++)
            {
                auto it_vals = std::get<mdf::DEDXTable>(it->second).T;
                auto it_ten = torch::from_blob(
                    it_vals.data(), n, torch::kFloat64);
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
                elements.push_back(element);
                element_id[name] = id;
                element_name.push_back(name);
                id++;
            }
        }

        inline void set_materials(const mdf::Materials &mdf_materials,
                                  const mdf::MaterialsDEDXData &dedx_data)
        {
            int id = 0;
            auto n_mats = mdf_materials.size();

            materials.reserve(n_mats);
            material_name.reserve(n_mats);

            material_density = torch::zeros(n_mats, default_ops);
            material_ZoA = torch::zeros(n_mats, default_ops);
            material_I = torch::zeros(n_mats, default_ops);

            material_density_effect.reserve(n_mats);

            for (const auto &[name, material] : mdf_materials)
            {
                auto [_, density, components] = material;
                int n = components.size();
                auto el_ids = torch::zeros(n, torch::kInt32);
                auto fracs = torch::zeros(n, default_ops);
                int iel = 0;
                for (const auto &[el, frac] : components)
                {
                    el_ids[iel] = element_id.at(el);
                    fracs[iel++] = frac;
                }

                materials.push_back(Material{el_ids, fracs / fracs.sum()});
                material_id[name] = id;
                material_name.push_back(name);

                material_density[id] = static_cast<Physics *>(this)->scale_density(density);

                const auto &coefs = std::get<mdf::DEDXMaterialCoefficients>(dedx_data.at(name));

                material_ZoA[id] = coefs.ZoA;
                material_I[id] = coefs.I;
                material_density_effect.push_back(coefs.density_effect);

                id++;
            }
        }

        inline TableCSn init_table_CSn()
        {
            auto shape = Shape(3);
            shape[0] = elements.size();
            shape[1] = dcs::NPR;
            shape[2] = table_K.numel();
            return torch::zeros(shape, default_ops);
        }

        inline void compute_cel_and_del(const TableCSn &del, const TableCSn &cel)
        {
            const int n = elements.size();
#pragma omp parallel for
            for (int el = 0; el < n; el++)
            {
                dcs::map_compute_integral(dcs_kernels, del[el], table_K, X_FRACTION, elements[el], mass, 180, false);
                dcs::map_compute_integral(dcs_kernels, cel[el], table_K, X_FRACTION, elements[el], mass, 180, true);
            }
        }

        inline void init_table_CSf()
        {
        }

        inline void set_dedx_tables(const mdf::MaterialsDEDXData &)
        {
            table_CSn = init_table_CSn();
            const auto table_cel = init_table_CSn();
            compute_cel_and_del(table_CSn, table_cel);

            int n = materials.size();
            for (int i = 0; i < n; i++)
            {
            }
        }

        inline Status initialise_physics(
            const mdf::Settings &mdf_settings, const mdf::MaterialsDEDXData &dedx_data)
        {
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings), dedx_data);

            if (!set_table_K(dedx_data))
                return false;

            set_dedx_tables(dedx_data);

            return true;
        }

    public:
        const DCSKernels dcs_kernels;
        const ParticleMass mass;
        const DecayLength ctau;

        PhysicsModel(DCSKernels dcs_kernels_, ParticleMass mass_, DecayLength ctau_)
            : dcs_kernels{dcs_kernels_}, mass{mass_}, ctau{ctau_} {}

        inline const AtomicElement &get_element(const ElementId id)
        {
            return elements.at(id);
        }
        inline const AtomicElement &get_element(const mdf::ElementName &name)
        {
            return elements.at(element_id.at(name));
        }
        inline const mdf::ElementName &get_element_name(const ElementId id)
        {
            return element_name.at(id);
        }

        inline const Material &get_material(const MaterialId id)
        {
            return materials.at(id);
        }
        inline const Material &get_material(const mdf::MaterialName &name)
        {
            return materials.at(material_id.at(name));
        }
        inline const mdf::MaterialName &get_material_name(const MaterialId id)
        {
            return material_name.at(id);
        }

        inline const MaterialsDensity &get_material_density()
        {
            return material_density;
        }
        inline const MaterialsZoA &get_material_ZoA()
        {
            return material_ZoA;
        }
        inline const MaterialsI &get_material_I()
        {
            return material_I;
        }
        inline const MaterialsDensityEffect &get_material_density_effect()
        {
            return material_density_effect;
        }

        inline const TableK &get_table_K()
        {
            return table_K;
        }
        inline const TableCSn &get_table_CSn()
        {
            return table_CSn;
        }
        inline const TableCSf &get_table_CSf()
        {
            return table_CSf;
        }

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
            return mass * 1E-3; // from MeV to GeV
        }

        inline TableK scale_table_K(const TableK &table_K_)
        {
            return table_K_ * 1E-3; // from MeV to GeV
        }

        inline MeanExcitation scale_excitation(const MeanExcitation &I)
        {
            return I * 1E-9; // from eV to GeV
        }

        inline MaterialDensity scale_density(const MaterialDensity &density)
        {
            return density * 1E+3; // from g/cm^3 to kg/m^3
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
