/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd. (roland.grinis@grinisrit.com)
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
#include "ghmc/utils/common.hh"

#include <torch/torch.h>

namespace ghmc::pms
{
    using namespace torch::indexing;

    using EnergyScale = Scalar;
    using DensityScale = Scalar;

    using Elements = std::vector<AtomicElement>;
    using ElementId = Index;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ELementIdsList = torch::Tensor;
    using ElementsFractions = torch::Tensor;
    using ElementNames = std::vector<mdf::ElementName>;

    struct Material
    {
        ELementIdsList element_ids;
        ElementsFractions fractions;
    };
    using Materials = std::vector<Material>;
    using MaterialId = Index;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using MaterialNames = std::vector<mdf::MaterialName>;

    using MaterialsDensity = torch::Tensor;
    using MaterialsZoA = torch::Tensor;
    using MaterialsI = torch::Tensor;
    using MaterialsDensityEffect = std::vector<MaterialDensityEffect>;

    using Shape = std::vector<int64_t>;
    using Table = torch::Tensor;

    using TableK = Table;                // Kinetic energy tabulations
    using TableCSn = Table;              // CS normalisation tabulations
    using TableCSf = std::vector<Table>; // CS fractions by material
    using TableCS = Table;               // CS for inelatic DELs
    using TableDE = Table;               // Average energy loss
    using TableX = Table;                // CSDA range
    using TableNIin = Table;             // Interaction lengths
    using IonisationMax = Table;         // Maximum tabulated a(E)
    using RadlossMax = Table;            // Maximum tabulated b(E)
    using TableKt = Table;               // Kinetic threshold for DELs
    using TableXt = Table;               // Fraction threshold for DELs
    using TableMu0 = Table;              // Angular cutoff for splitting of Coulomb Scattering
    using TableLb = Table;               // Interaction lengths for DEL Coulomb events
    using TableNIel = Table;             // EHS number of interaction lengths
    using TableMs1 = Table;              //Multiple scattering 1st moment

    inline const auto default_dtp = torch::dtype(torch::kFloat64);
    inline const auto default_ops = default_dtp.layout(torch::kStrided);

    struct CoulombWorkspace
    {
        dcs::TransportCoefs G;
        dcs::CMLorentz fCM;
        dcs::ScreeningFactors screening;
        dcs::InvLambdas invlambda;
        dcs::FSpins fspin;
        dcs::SoftScatter table_ms1;
    };

    template <typename Physics, typename DCSKernels>
    class PhysicsModel
    {
        using DELKernels = typename std::tuple_element<0, DCSKernels>::type;
        using TTKernels = typename std::tuple_element<1, DCSKernels>::type;

        using DCSBremsstrahlung = typename std::tuple_element<0, DELKernels>::type;
        using DCSPairProduction = typename std::tuple_element<1, DELKernels>::type;
        using DCSPhotonuclear = typename std::tuple_element<2, DELKernels>::type;
        using DCSIonisation = typename std::tuple_element<3, DELKernels>::type;

        using CoulombData = typename std::tuple_element<0, TTKernels>::type;
        using HardScattering = typename std::tuple_element<1, TTKernels>::type;
        using SoftScattering = typename std::tuple_element<2, TTKernels>::type;

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
        TableCS table_CS;
        TableDE table_dE;
        TableX table_X;
        TableNIin table_NI_in;
        IonisationMax table_a_max;
        RadlossMax table_b_max;
        TableKt table_Kt;
        TableXt table_Xt;
        TableMu0 table_Mu0;
        TableLb table_Lb;
        TableNIel table_NI_el;
        TableMs1 table_Ms1;

        inline utils::Status check_mass(mdf::MaterialsDEDXData &dedx_data)
        {
            for (const auto &[material, data] : dedx_data)
                if (static_cast<Physics *>(this)->scale_energy(std::get<ParticleMass>(data)) != mass)
                {
                    std::cerr << "Inconsistent particle mass in "
                                 "dedx data for "
                              << material << std::endl;
                    return false;
                }
            return true;
        }

        inline utils::Status set_table_K(mdf::MaterialsDEDXData &dedx_data)
        {
            auto data = dedx_data.begin();
            auto &vals = std::get<mdf::DEDXTable>(data->second).T;
            const int n = vals.size();
            auto tensor = torch::from_blob(vals.data(), n, default_ops);
            table_K = static_cast<Physics *>(this)->scale_energy(tensor);
            data++;
            for (auto &it = data; it != dedx_data.end(); it++)
            {
                auto &it_vals = std::get<mdf::DEDXTable>(it->second).T;
                auto it_ten = torch::from_blob(
                    it_vals.data(), n, default_ops);
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

        inline utils::Status perform_initial_checks(const mdf::Settings &mdf_settings,
                                                    mdf::MaterialsDEDXData &dedx_data)
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
                element.I = 1E-6 * static_cast<Physics *>(this)->scale_energy(element.I);
                elements.push_back(element);
                element_id[name] = id;
                element_name.push_back(name);
                id++;
            }
        }

        inline void set_materials(const mdf::Materials &mdf_materials,
                                  mdf::MaterialsDEDXData &dedx_data)
        {
            int id = 0;
            const int n_mats = mdf_materials.size();

            materials.reserve(n_mats);
            material_name.reserve(n_mats);

            material_density = torch::zeros(n_mats, default_ops);
            material_ZoA = torch::zeros(n_mats, default_ops);
            material_I = torch::zeros(n_mats, default_ops);

            material_density_effect.reserve(n_mats);

            for (const auto &[name, material] : mdf_materials)
            {
                auto [_, density, components] = material;
                const int n = components.size();
                auto el_ids = torch::zeros(n, torch::kInt64);
                auto fracs = torch::zeros(n, default_ops);
                int iel = 0;
                for (const auto &[el, frac] : components)
                {
                    el_ids[iel] = element_id.at(el);
                    fracs[iel++] = frac;
                }

                materials.push_back(Material{el_ids, fracs});
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

        inline Shape cs_shape(int nelems)
        {
            auto shape = Shape(3);
            shape[0] = nelems;
            shape[1] = dcs::NPR;
            shape[2] = table_K.numel();
            return shape;
        }

        inline void compute_cel_and_del(const TableCSn &del, const TableCSn &cel)
        {
            const int n = elements.size();
#pragma omp parallel for
            for (int el = 0; el < n; el++)
            {
                dcs::compute_dcs_integrals(
                    std::get<DELKernels>(dcs_kernels), del[el], table_K,
                    static_cast<Physics *>(this)->x_fraction(), elements[el], mass, 180, false);
                dcs::compute_dcs_integrals(
                    std::get<DELKernels>(dcs_kernels), cel[el], table_K,
                    static_cast<Physics *>(this)->x_fraction(), elements[el], mass, 180, true);
            }
        }

        inline void init_dedx_tables()
        {
            const int nmat = materials.size();
            const int nkin = table_K.numel();
            table_CSf = TableCSf(nmat);
            table_CS = torch::zeros({nmat, nkin}, default_ops);
            table_dE = torch::zeros_like(table_CS);
            table_X = torch::zeros_like(table_CS);
            table_NI_in = torch::zeros_like(table_CS);
            table_Kt = torch::zeros(nmat, default_ops);
            table_a_max = torch::zeros_like(table_Kt);
            table_b_max = torch::zeros_like(table_Kt);
        }

        inline dcs::ThresholdIndex compute_dedx_tables(
            MaterialId imat,
            const TableCSn &cel_table,
            mdf::DEDXTable &dedx_table)
        {
            const int nel = materials[imat].element_ids.numel();
            const int n = table_K.numel();

            table_CSf[imat] = materials[imat].fractions.view({nel, 1, 1}) *
                              table_CSn.index_select(0, materials[imat].element_ids);
            table_CS[imat] = table_CSf[imat].sum(0).sum(0);

            table_CSf[imat] *= torch::where(table_CS[imat] <= 0.0,
                                            torch::tensor(0.0, default_dtp), torch::tensor(1.0, default_dtp))
                                   .view({1, 1, n});
            table_CSf[imat] = table_CSf[imat].view({nel * dcs::NPR, n}).cumsum(0).view({nel, dcs::NPR, n}) /
                              torch::where(table_CS[imat] <= 0.0, torch::tensor(1.0, default_dtp), table_CS[imat]).view({1, 1, n});

            auto a = static_cast<Physics *>(this)->scale_dEdX(torch::from_blob(dedx_table.Ionisation.data(), n, default_ops));
            auto be_cel = dcs::compute_be_cel(
                static_cast<Physics *>(this)->scale_dEdX(torch::from_blob(dedx_table.brems.data(), n, default_ops)),
                static_cast<Physics *>(this)->scale_dEdX(torch::from_blob(dedx_table.pair.data(), n, default_ops)),
                static_cast<Physics *>(this)->scale_dEdX(torch::from_blob(dedx_table.photonuc.data(), n, default_ops)),
                a, torch::tensordot(materials[imat].fractions, cel_table.index_select(0, materials[imat].element_ids), 0, 0));

            table_dE[imat] = static_cast<Physics *>(this)->scale_dEdX(
                                 torch::from_blob(dedx_table.dEdX.data(), n, default_ops)) -
                             be_cel;
            table_X[imat] = 1.0 / table_dE[imat];
            table_NI_in[imat] = table_CS[imat] * table_X[imat];
            table_a_max[imat] = a[n - 1];
            table_b_max[imat] =
                (static_cast<Physics *>(this)->scale_dEdX(dedx_table.Radloss[n - 1]) - be_cel[n - 1]) /
                (mass + table_K[n - 1]);

            int ri = 0;
            Scalar cs0 = 0.0;
            Scalar *cs = table_CS[imat].data_ptr<Scalar>();
            for (ri = 0; ri < n; ri++)
                if ((cs0 = cs[ri]) > 0)
                    break;
            table_Kt[imat] = table_K[ri];
            Scalar *dE = table_dE[imat].data_ptr<Scalar>();
            Scalar *NI_in = table_NI_in[imat].data_ptr<Scalar>();
            for (int i = 0; i < ri; i++)
            {
                cs[i] = cs0;
                NI_in[i] = cs0 / dE[i];
            }
            return ri;
        }

        inline void compute_del_threshold(dcs::ThresholdIndex th_i)
        {
            table_Xt = torch::ones(cs_shape(elements.size()), default_ops);
            const int nel = elements.size();
            for (int el = 0; el < nel; el++)
                dcs::compute_fractional_thresholds(
                    std::get<DELKernels>(dcs_kernels), table_Xt[el], table_K,
                    static_cast<Physics *>(this)->x_fraction(), elements[el], mass, th_i);
        }

        inline void set_dedx_tables(mdf::MaterialsDEDXData &dedx_data)
        {
            table_CSn = torch::zeros(cs_shape(elements.size()), default_ops);
            const auto cel_table = torch::zeros_like(table_CSn);
            compute_cel_and_del(table_CSn, cel_table);

            init_dedx_tables();
            const int n = materials.size();
            dcs::ThresholdIndex th_i = 0;
            for (int imat = 0; imat < n; imat++)
                th_i = std::max(th_i, compute_dedx_tables(
                                          imat, cel_table,
                                          std::get<mdf::DEDXTable>(dedx_data.at(material_name.at(imat)))));

            compute_del_threshold(th_i);
        }

        inline CoulombWorkspace init_coulomb_parameters(const int nel, const int nmat)
        {
            const int nkin = table_K.numel();
            table_Mu0 = torch::zeros({nmat, nkin}, default_ops);
            table_Lb = torch::zeros_like(table_Mu0);
            table_NI_el = torch::zeros_like(table_Mu0);
            table_Ms1 = torch::zeros_like(table_Mu0);
            return CoulombWorkspace{
                torch::zeros({nel, nkin, 2}, default_ops),
                torch::zeros({nel, nkin, 2}, default_ops),
                torch::zeros({nel, nkin, 9}, default_ops),
                torch::zeros({nel, nkin}, default_ops),
                torch::zeros({nel, nkin}, default_ops),
                torch::zeros({nel, nkin}, default_ops)};
        }

        inline void compute_coulomb_data(const ElementId iel, CoulombWorkspace &cdata)
        {
            const auto &el = elements.at(iel);
            const auto &G = cdata.G[iel];
            const auto &fCM = cdata.fCM[iel];
            const auto &screen = cdata.screening[iel];
            Scalar *invlambda = cdata.invlambda[iel].data_ptr<Scalar>();
            Scalar *fspin = cdata.fspin[iel].data_ptr<Scalar>();
            Scalar *ms1 = cdata.table_ms1[iel].data_ptr<Scalar>();

            utils::for_eachi<Scalar>(
                table_K,
                [&](const int i, const auto &k) {
                    const auto &Gi = G[i];
                    const auto &screeni = screen[i];
                    invlambda[i] = std::get<CoulombData>(std::get<TTKernels>(dcs_kernels))(
                        fCM[i], screeni, fspin[i], k, el, mass);
                    dcs::coulomb_transport_coefficients(Gi, screeni, fspin[i], 1.);
                    ms1[i] = std::get<SoftScattering>(std::get<TTKernels>(dcs_kernels))(
                        k, el, mass);
                });
        }

        inline void compute_coulomb_scattering_tables(const MaterialId imat, const CoulombWorkspace &cdata)
        {
            const auto &elids = materials.at(imat).element_ids;
            const auto &fracs = materials.at(imat).fractions;
            const int nel = elids.numel();

            const auto G = cdata.G.index_select(0, elids).transpose(0, 1);
            const auto fCM = cdata.fCM.index_select(0, elids).transpose(0, 1);
            const auto screen = cdata.screening.index_select(0, elids).transpose(0, 1);
            const auto invlambda =
                (cdata.invlambda.index_select(0, elids) * materials.at(imat).fractions.view({nel, 1})).transpose(0, 1);
            const auto fspin = cdata.fspin.index_select(0, elids).transpose(0, 1);
            const auto ms1 =
                (cdata.table_ms1.index_select(0, elids) * fracs.view({nel, 1})).transpose(0, 1);

            Scalar *mu0 = table_Mu0[imat].data_ptr<Scalar>();
            Scalar *Lb = table_Lb[imat].data_ptr<Scalar>();
            Scalar *NI_el = table_NI_el[imat].data_ptr<Scalar>();
            Scalar *dE = table_dE[imat].data_ptr<Scalar>();
            Scalar *Ms1 = table_Ms1[imat].data_ptr<Scalar>();

            utils::for_eachi<Scalar>(
                table_K,
                [&](const int i, const auto &k) {
             
                    const auto Gi = G[i];
                    const auto fCMi = fCM[i];
                    const auto &screeni = screen[i];
                    const auto &invlambdai = invlambda[i].to(default_ops, false, true);
                    const auto &fspini = fspin[i].to(default_ops, false, true);
                    const auto &ms1i = ms1[i].to(default_ops, false, true);

                    const auto lb_h = std::get<HardScattering>(std::get<TTKernels>(dcs_kernels))(
                        mu0[i], Gi, fCMi, screeni, invlambdai, fspini);

                    Lb[i] = lb_h * k * (k + 2 * mass);
                    NI_el[i] = 1. / (dE[i] * lb_h);

                    Ms1[i] = dcs::compute_soft_scattering_for_material(
                        Gi, fCMi, screeni, invlambdai, fspini, ms1i, mu0[i]);
                });
        }

        inline void set_coulomb_parameters()
        {
            const int nel = elements.size();
            const int nmat = materials.size();
            auto cdata = init_coulomb_parameters(nel, nmat);

#pragma omp parallel for
            for (int iel = 0; iel < nel; iel++)
                compute_coulomb_data(iel, cdata);

            for (int imat = 0; imat < nmat; imat++)
                compute_coulomb_scattering_tables(imat, cdata);
        }

        inline utils::Status initialise_physics(
            const mdf::Settings &mdf_settings, mdf::MaterialsDEDXData &dedx_data)
        {
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings), dedx_data);

            if (!set_table_K(dedx_data))
                return false;

            set_dedx_tables(dedx_data);

            set_coulomb_parameters();

            return true;
        }

    public:
        const DCSKernels dcs_kernels;
        const ParticleMass mass;
        const DecayLength ctau;

        PhysicsModel(
            DCSKernels dcs_kernels_,
            ParticleMass mass_,
            DecayLength ctau_)
            : dcs_kernels{dcs_kernels_},
              mass{mass_},
              ctau{ctau_}
        {
        }

        inline const AtomicElement &get_element(const ElementId id) const
        {
            return elements.at(id);
        }
        inline const AtomicElement &get_element(const mdf::ElementName &name) const
        {
            return elements.at(element_id.at(name));
        }
        inline const mdf::ElementName &get_element_name(const ElementId id) const
        {
            return element_name.at(id);
        }

        inline const Material &get_material(const MaterialId id) const
        {
            return materials.at(id);
        }
        inline const Material &get_material(const mdf::MaterialName &name) const
        {
            return materials.at(material_id.at(name));
        }
        inline const mdf::MaterialName &get_material_name(const MaterialId id) const
        {
            return material_name.at(id);
        }

        inline const MaterialsDensity &get_material_density() const
        {
            return material_density;
        }
        inline const MaterialsZoA &get_material_ZoA() const
        {
            return material_ZoA;
        }
        inline const MaterialsI &get_material_I() const
        {
            return material_I;
        }
        inline const MaterialsDensityEffect &get_material_density_effect() const
        {
            return material_density_effect;
        }

        inline const TableK &get_table_K() const
        {
            return table_K;
        }
        inline const TableCSn &get_table_CSn() const
        {
            return table_CSn;
        }
        inline const TableCSf &get_table_CSf() const
        {
            return table_CSf;
        }
        inline const TableCS &get_table_CS() const
        {
            return table_CS;
        }
        inline const TableCS &get_table_dE() const
        {
            return table_dE;
        }
        inline const TableX &get_table_X() const
        {
            return table_X;
        }
        inline const TableNIin &get_table_NI_in() const
        {
            return table_NI_in;
        }
        inline const IonisationMax &get_table_a_max() const
        {
            return table_a_max;
        }
        inline const IonisationMax &get_table_b_max() const
        {
            return table_b_max;
        }
        inline const TableKt &get_table_Kt() const
        {
            return table_Kt;
        }
        inline const TableXt &get_table_Xt() const
        {
            return table_Xt;
        }
        inline const TableMu0 &get_table_Mu0() const
        {
            return table_Mu0;
        }
        inline const TableLb &get_table_Lb() const
        {
            return table_Lb;
        }
        inline const TableNIel &get_table_NI_el() const
        {
            return table_NI_el;
        }
        inline const TableMs1 &get_table_Ms1() const
        {
            return table_Ms1;
        }

        inline utils::Status
        load_physics_from(const mdf::Settings &mdf_settings,
                          mdf::MaterialsDEDXData &dedx_data)
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

        template <typename Energy>
        inline Energy scale_energy(const Energy &energy) const
        {
            return energy * 1E-3; // from MeV to GeV
        }

        template <typename Density>
        inline Density scale_density(const Density &density) const
        {
            return density * 1E+3; // from g/cm^3 to kg/m^3
        }

        template <typename Tables>
        inline Tables scale_dEdX(const Tables &tables) const
        {
            return tables * 1E-4; // from MeV cm^2/g to GeV m^2/kg
        }

        inline EnergyTransferMin x_fraction() const
        {
            return X_FRACTION;
        }

    public:
        MuonPhysics(DCSKernels dcs_kernels_,
                    ParticleMass mass_ = MUON_MASS,
                    DecayLength ctau_ = MUON_CTAU)
            : PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>(
                  dcs_kernels_, mass_, ctau_)
        {
        }
    };

    template <typename DCSKernels>
    struct TauPhysics : MuonPhysics<DCSKernels>
    {
        TauPhysics(DCSKernels dcs_kernels_,
                   ParticleMass mass_ = TAU_MASS,
                   DecayLength ctau_ = TAU_CTAU)
            : PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>(
                  dcs_kernels_, mass_, ctau_)
        {
        }
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
