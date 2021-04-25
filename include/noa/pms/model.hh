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

#include "noa/pms/mdf.hh"
#include "noa/pms/dcs_model.hh"
#include "noa/pms/constants.hh"
#include "noa/utils/common.hh"

#include <torch/torch.h>

namespace noa::pms {
    using namespace torch::indexing;

    using Elements = std::vector<AtomicElement<Scalar>>;
    using ElementId = Index;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ELementIdsList = torch::Tensor;
    using ElementsFractions = torch::Tensor;
    using ElementNames = std::vector<mdf::ElementName>;

    struct Material {
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

    using Table = dcs::Table;

    using TableK = Table;                // Kinetic energy tabulations
    using TableCSn = Table;              // CS normalisation tabulations
    using TableCSf = std::vector<Table>; // CS fractions by material
    using TableCS = Table;               // CS for inelastic DELs
    using TabledE = Table;               // Average energy loss
    using TabledECSDA = Table;           // Average energy loss in CSDA approx
    using TableX = Table;                // CSDA grammage range for energy loss
    using TableXCSDA = Table;            // CSDA grammage range for energy loss in CSDA approx.
    using TableT = Table;                // Total proper time
    using TableTCSDA = Table;            // Total proper time in CSDA approx.
    using TableNIin = Table;             // Interaction lengths
    using IonisationMax = Table;         // Maximum tabulated a(E)
    using RadlossMax = Table;            // Maximum tabulated b(E)
    using TableKt = Table;               // Kinetic threshold for DELs
    using TableXt = Table;               // Fraction threshold for DELs
    using TableMu0 = Table;              // Angular cutoff for splitting of Coulomb Scattering
    using TableLb = Table;               // Interaction lengths for DEL Coulomb events
    using TableNIel = Table;             // EHS number of interaction lengths
    using TableMs1 = Table;              // Multiple scattering 1st moment
    using TableLi = Table;               // Magnetic deflection momenta
    using DCSData = Table;               // DCS model coefficients

    struct CoulombWorkspace {
        dcs::TransportCoefs G;
        dcs::CMLorentz fCM;
        dcs::ScreeningFactors screening;
        dcs::InvLambdas invlambda;
        dcs::FSpins fspin;
        dcs::SoftScatter table_ms1;
    };

    template<typename Physics, typename DCSKernels>
    class PhysicsModel {
        using DELKernels = typename std::tuple_element<0, DCSKernels>::type;
        using TTKernels = typename std::tuple_element<1, DCSKernels>::type;

        using CoulombData = typename std::tuple_element<0, TTKernels>::type;
        using CoulombTransport = typename std::tuple_element<1, TTKernels>::type;
        using HardScattering = typename std::tuple_element<2, TTKernels>::type;
        using SoftScattering = typename std::tuple_element<3, TTKernels>::type;

        c10::TensorOptions tensor_ops = torch::dtype(torch::kDouble).layout(torch::kStrided);

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
        TabledE table_dE;
        TabledECSDA table_dE_CSDA;
        TableX table_X;
        TableXCSDA table_X_CSDA;
        TableT table_T;
        TableTCSDA table_T_CSDA;
        TableNIin table_NI_in;
        IonisationMax table_a_max;
        RadlossMax table_b_max;
        TableKt table_Kt;
        TableXt table_Xt;
        TableMu0 table_Mu0;
        TableLb table_Lb;
        TableNIel table_NI_el;
        TableMs1 table_Ms1;
        TableLi table_Li;
        DCSData dcs_data;

        TableCSn cel_table;
        CoulombWorkspace coulomb_workspace;

        inline utils::Status check_mass(mdf::MaterialsDEDXData &dedx_data) {
            for (const auto &[material, data] : dedx_data)
                if (static_cast<Physics *>(this)->scale_energy(std::get<ParticleMass>(data)) != mass) {
                    std::cerr << "Inconsistent particle mass in "
                                 "dedx data for "
                              << material << std::endl;
                    return false;
                }
            return true;
        }

        inline utils::Status check_ZoA(
                const mdf::Settings &mdf_settings, const mdf::MaterialsDEDXData &dedx_data) {
            auto mdf_elements = std::get<mdf::Elements>(mdf_settings);
            auto mdf_materials = std::get<mdf::Materials>(mdf_settings);
            for (const auto &[material, data] : dedx_data) {
                auto coefs = std::get<mdf::DEDXMaterialCoefficients>(data);
                Scalar ZoA = 0.0;
                for (const auto &[elmt, frac] :
                        std::get<mdf::MaterialComponents>(mdf_materials.at(material)))
                    ZoA += frac * mdf_elements.at(elmt).Z / mdf_elements.at(elmt).A;
                if ((std::abs(ZoA - coefs.ZoA) > utils::TOLERANCE))
                    return false;
            }
            return true;
        }

        inline utils::Status set_table_K(mdf::MaterialsDEDXData &dedx_data) {
            auto data = dedx_data.begin();
            auto &values = std::get<mdf::DEDXTable>(data->second).T;
            const int nkin = values.size();
            auto tensor = torch::from_blob(values.data(), nkin, tensor_ops);
            table_K = static_cast<Physics *>(this)->scale_energy(tensor);
            data++;
            for (auto &it = data; it != dedx_data.end(); it++) {
                auto &it_vals = std::get<mdf::DEDXTable>(it->second).T;
                auto it_ten = torch::from_blob(
                        it_vals.data(), nkin, tensor_ops);
                if (!torch::equal(tensor, it_ten)) {
                    std::cerr
                            << "Inconsistent kinetic energy values for "
                            << it->first << std::endl;
                    return false;
                }
            }
            return true;
        }

        inline utils::Status perform_initial_checks(const mdf::Settings &mdf_settings,
                                                    mdf::MaterialsDEDXData &dedx_data) {
            if (!check_ZoA(mdf_settings, dedx_data))
                return false;
            if (!check_mass(dedx_data))
                return false;
            return true;
        }

        inline void set_elements(const mdf::Elements &mdf_elements) {
            int id = 0;
            elements.reserve(mdf_elements.size());
            for (auto[name, element] : mdf_elements) {
                element.I = 1E-6 * static_cast<Physics *>(this)->scale_energy(element.I);
                elements.push_back(element);
                element_id[name] = id;
                element_name.push_back(name);
                id++;
            }
        }

        inline void set_materials(const mdf::Materials &mdf_materials,
                                  mdf::MaterialsDEDXData &dedx_data) {
            int id = 0;
            const int nmat = mdf_materials.size();

            materials.reserve(nmat);
            material_name.reserve(nmat);

            material_density = torch::zeros(nmat, tensor_ops);
            material_ZoA = torch::zeros(nmat, tensor_ops);
            material_I = torch::zeros(nmat, tensor_ops);

            material_density_effect.reserve(nmat);

            for (const auto &[name, material] : mdf_materials) {
                auto[_, density, components] = material;
                const int n = components.size();
                auto el_ids = torch::zeros(n, torch::kInt64);
                auto fracs = torch::zeros(n, tensor_ops);
                int iel = 0;
                for (const auto &[el, frac] : components) {
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

        inline void init_dcs_data(const int nel, const int nkin) {
            dcs_data = torch::zeros({nel, dcs::NPR - 1, nkin, dcs::NDM}, tensor_ops);
        }

        inline void init_per_element_data(const int nel) {
            const int nkin = table_K.numel();
            table_CSn = torch::zeros({nel, dcs::NPR, nkin}, tensor_ops);
            cel_table = torch::zeros_like(table_CSn);
            coulomb_workspace = CoulombWorkspace{
                    torch::zeros({nel, nkin, 2}, tensor_ops),
                    torch::zeros({nel, nkin, 2}, tensor_ops),
                    torch::zeros({nel, nkin, dcs::NSF}, tensor_ops),
                    torch::zeros({nel, nkin}, tensor_ops),
                    torch::zeros({nel, nkin}, tensor_ops),
                    torch::zeros({nel, nkin}, tensor_ops)};
        }

        inline void compute_coulomb_data(const ElementId iel) {
            const auto &element = elements.at(iel);
            const auto &screen = coulomb_workspace.screening[iel];
            const auto &fspin = coulomb_workspace.fspin[iel];

            std::get<CoulombData>(std::get<TTKernels>(dcs_kernels))(
                    coulomb_workspace.fCM[iel],
                    screen, fspin,
                    coulomb_workspace.invlambda[iel],
                    table_K, element, mass);

            std::get<CoulombTransport>(std::get<TTKernels>(dcs_kernels))(
                    coulomb_workspace.G[iel],
                    screen, fspin,
                    torch::tensor(1.0, tensor_ops));

            std::get<SoftScattering>(std::get<TTKernels>(dcs_kernels))(
                    coulomb_workspace.table_ms1[iel],
                    table_K, element, mass);
        }

        inline void compute_per_element_data() {
            const int nel = num_elements();
            const auto &model_K = table_K.index(
                    {table_K >= static_cast<Physics *>(this)->dcs_model_min_energy()});

            init_per_element_data(nel);
            init_dcs_data(nel, model_K.numel());

#pragma omp parallel for
            for (int iel = 0; iel < nel; iel++) {
                dcs::compute_dcs_integrals(
                        std::get<DELKernels>(dcs_kernels), table_CSn[iel], table_K,
                        static_cast<Physics *>(this)->x_fraction(), elements[iel], mass, 180, false);
                dcs::compute_dcs_integrals(
                        std::get<DELKernels>(dcs_kernels), cel_table[iel], table_K,
                        static_cast<Physics *>(this)->x_fraction(), elements[iel], mass, 180, true);

                dcs::compute_dcs_model(
                        std::get<DELKernels>(dcs_kernels),
                        dcs_data[iel],
                        model_K,
                        static_cast<Physics *>(this)->x_fraction(),
                        static_cast<Physics *>(this)->dcs_model_max_fraction(),
                        elements[iel],
                        mass);

                compute_coulomb_data(iel);
            }
        }

        inline void init_dedx_tables(const int nmat) {
            table_CSf = TableCSf(nmat);
            table_CS = torch::zeros({nmat, table_K.numel()}, tensor_ops);
            table_dE = torch::zeros_like(table_CS);
            table_dE_CSDA = torch::zeros_like(table_CS);
            table_NI_in = torch::zeros_like(table_CS);
            table_Kt = torch::zeros(nmat, tensor_ops);
            table_a_max = torch::zeros_like(table_Kt);
            table_b_max = torch::zeros_like(table_Kt);
        }

        inline dcs::ThresholdIndex compute_dedx_tables(
                MaterialId imat, mdf::DEDXTable &dedx_table) {
            const int nel = materials[imat].element_ids.numel();
            const int nkin = table_K.numel();

            table_CSf[imat] = materials[imat].fractions.view({nel, 1, 1}) *
                              table_CSn.index_select(0, materials[imat].element_ids);
            table_CS[imat] = table_CSf[imat].sum(0).sum(0);

            table_CSf[imat] *= torch::where(table_CS[imat] <= 0.0,
                                            torch::tensor(0.0, tensor_ops), torch::tensor(1.0, tensor_ops))
                    .view({1, 1, nkin});
            table_CSf[imat] = table_CSf[imat].view({nel * dcs::NPR, nkin}).cumsum(0).view({nel, dcs::NPR, nkin}) /
                              torch::where(table_CS[imat] <= 0.0, torch::tensor(1.0, tensor_ops),
                                           table_CS[imat]).view({1, 1, nkin});

            auto ionisation = static_cast<Physics *>(this)->scale_dEdX(
                    torch::from_blob(dedx_table.Ionisation.data(), nkin, tensor_ops));
            auto be_cel = dcs::compute_be_cel(
                    static_cast<Physics *>(this)->scale_dEdX(
                            torch::from_blob(dedx_table.brems.data(), nkin, tensor_ops)),
                    static_cast<Physics *>(this)->scale_dEdX(
                            torch::from_blob(dedx_table.pair.data(), nkin, tensor_ops)),
                    static_cast<Physics *>(this)->scale_dEdX(
                            torch::from_blob(dedx_table.photonuc.data(), nkin, tensor_ops)),
                    ionisation,
                    torch::tensordot(materials[imat].fractions, cel_table.index_select(0, materials[imat].element_ids),
                                     0, 0));

            table_dE_CSDA[imat] = static_cast<Physics *>(this)->scale_dEdX(
                    torch::from_blob(dedx_table.dEdX.data(), nkin, tensor_ops));
            table_dE[imat] = table_dE_CSDA[imat] - be_cel;
            table_NI_in[imat] = table_CS[imat] / table_dE[imat];
            table_a_max[imat] = ionisation[nkin - 1];
            table_b_max[imat] =
                    (static_cast<Physics *>(this)->scale_dEdX(dedx_table.Radloss[nkin - 1]) - be_cel[nkin - 1]) /
                    (mass + table_K[nkin - 1]);

            int ri = dcs::compute_kinetic_threshold(table_CS[imat], table_dE[imat], table_NI_in[imat]);
            table_Kt[imat] = table_K[ri];

            return ri;
        }

        inline void compute_del_threshold(dcs::ThresholdIndex th_i) {
            table_Xt = torch::ones_like(table_CSn);
            const int nel = num_elements();
            for (int iel = 0; iel < nel; iel++)
                dcs::compute_fractional_thresholds(
                        std::get<DELKernels>(dcs_kernels), table_Xt[iel], table_K,
                        static_cast<Physics *>(this)->x_fraction(), elements[iel], mass, th_i);
        }

        inline void set_dedx_tables(mdf::MaterialsDEDXData &dedx_data) {
            const int nmat = num_materials();
            init_dedx_tables(nmat);
            dcs::ThresholdIndex th_i = 0;
            for (int imat = 0; imat < nmat; imat++)
                th_i = std::max(
                        th_i,
                        compute_dedx_tables(imat,
                                            std::get<mdf::DEDXTable>(dedx_data.at(material_name.at(imat)))));
            cel_table = torch::Tensor{}; // drop data in cel_table
            compute_del_threshold(th_i);
        }

        inline void init_coulomb_parameters() {
            table_Mu0 = torch::zeros_like(table_dE);
            table_Lb = torch::zeros_like(table_Mu0);
            table_NI_el = torch::zeros_like(table_Mu0);
            table_Ms1 = torch::zeros_like(table_Mu0);
        }

        inline void compute_coulomb_scattering_tables(const MaterialId imat) {
            const auto &elids = materials.at(imat).element_ids;
            const auto &fracs = materials.at(imat).fractions;
            const int nel = elids.numel();

            const auto G = coulomb_workspace.G.index_select(0, elids);
            const auto fCM = coulomb_workspace.fCM.index_select(0, elids);
            const auto screen = coulomb_workspace.screening.index_select(0, elids);
            const auto invlambda =
                    coulomb_workspace.invlambda.index_select(0, elids) * fracs.view({nel, 1});
            const auto fspin = coulomb_workspace.fspin.index_select(0, elids);
            const auto ms1 = coulomb_workspace.table_ms1.index_select(0, elids) * fracs.view({nel, 1});

            const auto &mu0 = table_Mu0[imat];
            const auto lb_h = torch::zeros_like(mu0);

            std::get<HardScattering>(std::get<TTKernels>(dcs_kernels))(
                    mu0, lb_h, G, fCM, screen, invlambda, fspin);

            table_Lb[imat] = lb_h * table_K * (table_K + 2 * mass);
            table_NI_el[imat] = 1. / (table_dE[imat] * lb_h);

            for (int iel = 0; iel < nel; iel++) {
                const auto &Gi = G[iel];
                std::get<CoulombTransport>(std::get<TTKernels>(dcs_kernels))(
                        Gi, screen[iel], fspin[iel], mu0);

                const auto &fCMi = fCM[iel];

                table_Ms1[imat] += ms1[iel] +
                                   invlambda[iel] * Gi.index({Ellipsis, 1}) *
                                   (1. / (fCMi.index({Ellipsis, 0}) * (1. + fCMi.index({Ellipsis, 1})))).pow(2);
            }
        }

        inline void set_coulomb_parameters() {
            const int nmat = num_materials();
            init_coulomb_parameters();

            for (int imat = 0; imat < nmat; imat++)
                compute_coulomb_scattering_tables(imat);

            coulomb_workspace = CoulombWorkspace{}; // drop CoulombWorkspace data
        }

        inline void init_cel_integrals(const int nmat) {
            table_X = torch::zeros_like(table_dE);
            table_X_CSDA = torch::zeros_like(table_dE);
            table_T = torch::zeros_like(table_dE);
            table_T_CSDA = torch::zeros_like(table_dE);
            table_Li = torch::zeros({nmat, table_K.numel(), dcs::NLAR}, tensor_ops);
        }

        inline void set_cel_integrals() {
            const int nmat = num_materials();
            init_cel_integrals(nmat);

            const Scalar I0 = dcs::compute_momentum_integral(table_K[0].item<Scalar>(), mass);
            for (int imat = 0; imat < nmat; imat++) {
                dcs::compute_cel_grammage_integral(table_X[imat], table_dE[imat], table_K);
                dcs::compute_cel_grammage_integral(table_X_CSDA[imat], table_dE_CSDA[imat], table_K);
                dcs::compute_time_integral(table_T[imat], table_X[imat], table_K, mass, I0);
                dcs::compute_time_integral(table_T_CSDA[imat], table_X_CSDA[imat], table_K, mass, I0);
                dcs::compute_kinetic_integral(table_NI_el[imat], table_K);
                dcs::compute_kinetic_integral(table_NI_in[imat], table_K);
                dcs::compute_csda_magnetic_transport(table_Li[imat], table_T_CSDA[imat], table_X_CSDA[imat], mass,
                                                     static_cast<Physics *>(this)->larmor_factor());
            }
        }

        inline utils::Status initialise_physics(
                const mdf::Settings &mdf_settings, mdf::MaterialsDEDXData &dedx_data) {
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings), dedx_data);

            if (!set_table_K(dedx_data))
                return false;

            compute_per_element_data();

            set_dedx_tables(dedx_data);
            set_coulomb_parameters();
            set_cel_integrals();

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
                  ctau{ctau_} {
        }

        inline const AtomicElement<Scalar> &get_element(const ElementId id) const {
            return elements.at(id);
        }

        inline const AtomicElement<Scalar> &get_element(const mdf::ElementName &name) const {
            return elements.at(element_id.at(name));
        }

        inline const mdf::ElementName &get_element_name(const ElementId id) const {
            return element_name.at(id);
        }
        
        inline int num_elements() const {
            return elements.size();
        }

        inline const Material &get_material(const MaterialId id) const {
            return materials.at(id);
        }

        inline const Material &get_material(const mdf::MaterialName &name) const {
            return materials.at(material_id.at(name));
        }

        inline const mdf::MaterialName &get_material_name(const MaterialId id) const {
            return material_name.at(id);
        }

        inline int num_materials() const {
            return materials.size();
        }

        inline const MaterialsDensity &get_material_density() const {
            return material_density;
        }

        inline const MaterialsZoA &get_material_ZoA() const {
            return material_ZoA;
        }

        inline const MaterialsI &get_material_I() const {
            return material_I;
        }

        inline const MaterialsDensityEffect &get_material_density_effect() const {
            return material_density_effect;
        }

        inline const TableK &get_table_K() const {
            return table_K;
        }

        inline const TableCSn &get_table_CSn() const {
            return table_CSn;
        }

        inline const TableCSf &get_table_CSf() const {
            return table_CSf;
        }

        inline const TableCS &get_table_CS() const {
            return table_CS;
        }

        inline const TableCS &get_table_dE() const {
            return table_dE;
        }

        inline const TableCS &get_table_dE_CSDA() const {
            return table_dE_CSDA;
        }

        inline const TableX &get_table_X() const {
            return table_X;
        }

        inline const TableX &get_table_X_CSDA() const {
            return table_X;
        }

        inline const TableT &get_table_T() const {
            return table_T;
        }

        inline const TableT &get_table_T_CSDA() const {
            return table_T;
        }

        inline const TableNIin &get_table_NI_in() const {
            return table_NI_in;
        }

        inline const IonisationMax &get_table_a_max() const {
            return table_a_max;
        }

        inline const IonisationMax &get_table_b_max() const {
            return table_b_max;
        }

        inline const TableKt &get_table_Kt() const {
            return table_Kt;
        }

        inline const TableXt &get_table_Xt() const {
            return table_Xt;
        }

        inline const TableMu0 &get_table_Mu0() const {
            return table_Mu0;
        }

        inline const TableLb &get_table_Lb() const {
            return table_Lb;
        }

        inline const TableNIel &get_table_NI_el() const {
            return table_NI_el;
        }

        inline const TableMs1 &get_table_Ms1() const {
            return table_Ms1;
        }

        inline const TableLb &get_table_Li() const {
            return table_Li;
        }

        inline const DCSData &get_dcs_data() const {
            return dcs_data;
        }

        inline utils::Status
        load_physics_from(const mdf::Settings &mdf_settings,
                          mdf::MaterialsDEDXData &dedx_data) {
            if (!perform_initial_checks(mdf_settings, dedx_data))
                return false;
            if (!initialise_physics(mdf_settings, dedx_data))
                return false;
            return true;
        }
    };

    template<typename DCSKernels>
    class MuonPhysics : public PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels> {
        friend class PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>;

        template<typename Energy_t>
        inline Energy_t scale_energy(const Energy_t &energy) const {
            return energy * 1E-3; // from MeV to GeV
        }

        template<typename Density_t>
        inline Density_t scale_density(const Density_t &density) const {
            return density * 1E+3; // from g/cm^3 to kg/m^3
        }

        template<typename Table_t>
        inline Table_t scale_dEdX(const Table_t &tables) const {
            return tables * 1E-4; // from MeV cm^2/g to GeV m^2/kg
        }

        [[nodiscard]] inline EnergyTransfer x_fraction() const {
            return X_FRACTION;
        }

        [[nodiscard]] inline EnergyTransfer dcs_model_max_fraction() const {
            return DCS_MODEL_MAX_FRACTION;
        }

        [[nodiscard]] inline Energy dcs_model_min_energy() const {
            return DCS_MODEL_MIN_KINETIC;
        }

        [[nodiscard]] inline LarmorFactor larmor_factor() const {
            return LARMOR_FACTOR;
        }

    public:
        explicit MuonPhysics(DCSKernels dcs_kernels_,
                             ParticleMass mass_ = MUON_MASS,
                             DecayLength ctau_ = MUON_CTAU)
                : PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>(
                dcs_kernels_, mass_, ctau_) {
        }
    };

    template<typename DCSKernels>
    struct TauPhysics : MuonPhysics<DCSKernels> {
        explicit TauPhysics(DCSKernels dcs_kernels_,
                   ParticleMass mass_ = TAU_MASS,
                   DecayLength ctau_ = TAU_CTAU)
                : PhysicsModel<MuonPhysics<DCSKernels>, DCSKernels>(
                dcs_kernels_, mass_, ctau_) {
        }
    };

    template<typename PumasPhysics, typename DCSKernels>
    inline std::optional<PumasPhysics> load_pumas_physics_from(
            const mdf::ParticleName &particle_name, const mdf::MDFFilePath &mdf,
            const mdf::DEDXFolderPath &dedx, const DCSKernels &dcs_kernels) {
        if (!utils::check_path_exists(mdf))
            return std::nullopt;
        if (!utils::check_path_exists(dedx))
            return std::nullopt;

        auto mdf_settings = mdf::parse_settings(mdf::pms, mdf);
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

    template<typename DCSKernels>
    inline std::optional<MuonPhysics<DCSKernels>> load_muon_physics_from(
            const mdf::MDFFilePath &mdf, const mdf::DEDXFolderPath &dedx, const DCSKernels &dcs_kernels) {
        return load_pumas_physics_from<MuonPhysics<DCSKernels>, DCSKernels>(mdf::Muon, mdf, dedx, dcs_kernels);
    }

} // namespace noa::pms
