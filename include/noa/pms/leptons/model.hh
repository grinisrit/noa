/**
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

#include "noa/pms/leptons/mdf.hh"
#include "noa/pms/leptons/physics.hh"
#include "noa/pms/leptons/dcs.hh"

#include <torch/torch.h>

namespace noa::pms::leptons {

    using namespace torch::indexing;

    using ElementId = Index;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ElementNames = std::vector<mdf::ElementName>;

    using ElementIdsList = torch::Tensor;
    using ElementsFractions = torch::Tensor;

    struct Material {
        ElementIdsList element_ids;
        ElementsFractions fractions;
        MaterialDensity density;
    };

    using MaterialId = Index;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using MaterialNames = std::vector<mdf::MaterialName>;

    using MaterialsRelativeElectronicDensity = torch::Tensor;
    using MaterialsMeanExcitationEnergy = torch::Tensor;
    using MaterialsDensityEffect = std::vector<MaterialDensityEffect<Scalar>>;

    using TableK = Tabulation;                // Kinetic energy tabulations
    using TableCSn = Tabulation;              // CS normalisation tabulations
    using TableCSf = std::vector<Tabulation>; // CS fractions by material
    using TableCS = Tabulation;               // CS for inelastic DELs
    using TabledE = Tabulation;               // Average energy loss
    using TabledECSDA = Tabulation;           // Average energy loss in CSDA approx
    using TableX = Tabulation;                // CSDA grammage range for energy loss
    using TableXCSDA = Tabulation;            // CSDA grammage range for energy loss in CSDA approx.
    using TableT = Tabulation;                // Total proper time
    using TableTCSDA = Tabulation;            // Total proper time in CSDA approx.
    using TableNIin = Tabulation;             // Interaction lengths
    using IonisationMax = Tabulation;         // Maximum tabulated a(E)
    using RadlossMax = Tabulation;            // Maximum tabulated b(E)
    using TableKt = Tabulation;               // Kinetic threshold for DELs
    using TableXt = Tabulation;               // Fraction threshold for DELs
    using TableMu0 = Tabulation;              // Angular cutoff for splitting of Coulomb Scattering
    using TableLb = Tabulation;               // Interaction lengths for DEL Coulomb events
    using TableNIel = Tabulation;             // EHS number of interaction lengths
    using TableMs1 = Tabulation;              // Multiple scattering 1st moment
    using TableLi = Tabulation;               // Magnetic deflection momenta
    using DCSData = Tabulation;               // DCS model coefficients
    using ThresholdIndex = Index;


    struct CoulombWorkspace {
        dcs::TransportCoefs G;
        dcs::CMLorentz fCM;
        dcs::ScreeningFactors screening;
        dcs::InvLambdas invlambda;
        dcs::FSpins fspin;
        dcs::SoftScatter table_ms1;
    };

    template<typename Physics>
    class Model {
        using Elements = std::vector<AtomicElement>;
        using Materials = std::vector<Material>;

        Elements elements;
        ElementIds element_id;
        ElementNames element_name;

        Materials materials;
        MaterialIds material_id;
        MaterialNames material_name;

        // TODO: Composite materials

        inline AtomicElement process_element_(const AtomicElement &element) {
            return static_cast<Physics *>(this)->process_element(element);
        }

        inline Material process_material_(
                const MaterialDensity &density,
                const mdf::MaterialComponents &components) {
            const Index n = components.size();
            const auto element_ids = torch::zeros(n, torch::kInt32);
            const auto fractions = torch::zeros(n, torch::dtype(c10::CppTypeToScalarType<Scalar>{}));
            Index iel = 0;
            for (const auto &[el, frac]: components) {
                element_ids[iel] = get_element_id(el);
                fractions[iel++] = frac;
            }
            return static_cast<Physics *>(this)->process_material(
                    Material{element_ids, fractions, density});
        }

    public:
        inline const AtomicElement &get_element(const ElementId id) const {
            return elements.at(id);
        }

        inline const AtomicElement &get_element(const mdf::ElementName &name) const {
            return elements.at(element_id.at(name));
        }

        inline const mdf::ElementName &get_element_name(const ElementId id) const {
            return element_name.at(id);
        }

        inline const ElementId &get_element_id(const mdf::ElementName &name) const {
            return element_id.at(name);
        }

        inline Index num_elements() const {
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

        inline const MaterialId &get_material_id(const mdf::MaterialName &name) const {
            return material_id.at(name);
        }

        inline Index num_materials() const {
            return materials.size();
        }

        inline void set_elements(const mdf::Elements &mdf_elements) {
            Index id = 0;
            elements.reserve(mdf_elements.size());
            for (const auto &[name, element]: mdf_elements) {
                elements.push_back(process_element_(element));
                element_id[name] = id;
                element_name.push_back(name);
                id++;
            }
        }

        inline void set_materials(const mdf::Materials &mdf_materials) {
            Index id = 0;
            const Index nmat = mdf_materials.size();

            materials.reserve(nmat);
            material_name.reserve(nmat);

            for (const auto &[name, material]: mdf_materials) {
                const auto &[_, density, components] = material;
                materials.push_back(process_material_(density, components));
                material_id[name] = id;
                material_name.push_back(name);
                id++;
            }
        }
    };


    class MuonPhysics : public Model<MuonPhysics> {

        friend class Model<MuonPhysics>;

        MaterialsRelativeElectronicDensity material_ZoA;
        MaterialsMeanExcitationEnergy material_I;
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

        // TODO: parametrise MuonPhysics by device when CUDA support available
        inline c10::TensorOptions tensor_ops() const {
            return torch::dtype(c10::CppTypeToScalarType<Scalar>{}).layout(torch::kStrided);
        }

        inline AtomicElement process_element(const AtomicElement &element) const {
            return AtomicElement{element.A, 1E-6 * dcs::ENERGY_SCALE * element.I, element.Z};
        }

        inline Material process_material(
                const Material &material) const {
            return Material{
                    material.element_ids,
                    material.fractions.to(tensor_ops()),
                    dcs::DENSITY_SCALE * material.density};
        }

        inline utils::Status process_dedx_data_header(const mdf::MaterialsDEDXData &dedx_data) {
            if (!mdf::check_particle_mass(MUON_MASS / dcs::ENERGY_SCALE, dedx_data))
                return false;

            const Index nmat = num_materials();
            material_ZoA = torch::zeros(nmat, tensor_ops());
            material_I = torch::zeros(nmat, tensor_ops());

            for (Index i = 0; i < nmat; i++) {
                const auto &coefs = std::get<mdf::DEDXMaterialCoefficients>(
                        dedx_data.at(get_material_name(i)));
                material_ZoA[i] = coefs.ZoA;
                material_I[i] = coefs.I;
                material_density_effect.push_back(coefs.density_effect);
            }
            return true;
        }

        inline utils::Status set_table_K(mdf::MaterialsDEDXData &dedx_data) {
            auto data = dedx_data.begin();
            auto &values = std::get<mdf::DEDXTable>(data->second).T;
            const Index nkin = values.size();
            const auto tensor = torch::from_blob(values.data(), nkin, torch::kDouble);
            table_K = tensor.to(tensor_ops()) * dcs::ENERGY_SCALE;
            data++;
            for (auto &it = data; it != dedx_data.end(); it++) {
                auto &it_vals = std::get<mdf::DEDXTable>(it->second).T;
                const auto it_ten = torch::from_blob(
                        it_vals.data(), nkin, torch::kDouble);
                if (!torch::equal(tensor, it_ten)) {
                    std::cerr
                            << "Inconsistent kinetic energy values for "
                            << it->first << std::endl;
                    return false;
                }
            }
            return true;
        }

        inline void init_dcs_data(const Index nel, const Index nkin) {
            dcs_data = torch::zeros({nel, dcs::NPR - 1, nkin, dcs::NDM}, tensor_ops());
        }

        inline void init_per_element_data(const Index nel) {
            const Index nkin = table_K.numel();
            table_CSn = torch::zeros({nel, dcs::NPR, nkin}, tensor_ops());
            cel_table = torch::zeros_like(table_CSn);
            table_Xt = torch::ones_like(table_CSn);
            coulomb_workspace = CoulombWorkspace{
                    torch::zeros({nel, nkin, 2}, tensor_ops()),
                    torch::zeros({nel, nkin, 2}, tensor_ops()),
                    torch::zeros({nel, nkin, dcs::NSF}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops())};
        }

        inline void init_dedx_tables(const Index nmat) {
            table_CSf = TableCSf(nmat);
            table_CS = torch::zeros({nmat, table_K.numel()}, tensor_ops());
            table_dE = torch::zeros_like(table_CS);
            table_dE_CSDA = torch::zeros_like(table_CS);
            table_NI_in = torch::zeros_like(table_CS);
            table_Kt = torch::zeros(nmat, tensor_ops());
            table_a_max = torch::zeros_like(table_Kt);
            table_b_max = torch::zeros_like(table_Kt);
        }

        inline void init_coulomb_parameters() {
            table_Mu0 = torch::zeros_like(table_dE);
            table_Lb = torch::zeros_like(table_Mu0);
            table_NI_el = torch::zeros_like(table_Mu0);
            table_Ms1 = torch::zeros_like(table_Mu0);
        }

        inline void init_cel_integrals(const Index nmat) {
            table_X = torch::zeros_like(table_dE);
            table_X_CSDA = torch::zeros_like(table_dE);
            table_T = torch::zeros_like(table_dE);
            table_T_CSDA = torch::zeros_like(table_dE);
            table_Li = torch::zeros({nmat, table_K.numel(), dcs::NLAR}, tensor_ops());
        }

        template<typename EnergyIntegrand>
        inline void compute_recoil_energy_integrals(
                const Tabulation &result,
                const EnergyIntegrand &integrand,
                const AtomicElement &element) {

            dcs::vmap_integral(
                    dcs::recoil_integral(dcs::bremsstrahlung, integrand))
                    (result[0], table_K, dcs::X_FRACTION, element, MUON_MASS, 180);
            dcs::vmap_integral(
                    dcs::recoil_integral(dcs::pair_production, integrand))
                    (result[1], table_K, dcs::X_FRACTION, element, MUON_MASS, 180);
            dcs::vmap_integral(
                    dcs::recoil_integral(dcs::photonuclear, integrand))
                    (result[2], table_K, dcs::X_FRACTION, element, MUON_MASS, 180);
            dcs::vmap_integral(
                    dcs::recoil_integral(dcs::ionisation, integrand))
                    (result[3], table_K, dcs::X_FRACTION, element, MUON_MASS, 180);
        }

        inline void compute_dcs_model(
                const Tabulation &result,
                const TableK &high_kinetic_energies,
                const AtomicElement &element) {
            polynomial_model_fit(dcs::bremsstrahlung,
                                 result[0], high_kinetic_energies, dcs::X_FRACTION,
                                 dcs::DCS_MODEL_MAX_FRACTION,
                                 element, MUON_MASS);
            polynomial_model_fit(dcs::pair_production,
                                 result[1], high_kinetic_energies, dcs::X_FRACTION,
                                 dcs::DCS_MODEL_MAX_FRACTION,
                                 element, MUON_MASS);
            polynomial_model_fit(dcs::photonuclear,
                                 result[2], high_kinetic_energies, dcs::X_FRACTION,
                                 dcs::DCS_MODEL_MAX_FRACTION,
                                 element, MUON_MASS);
        }

        inline void compute_coulomb_data(const AtomicElement &element, const ElementId iel) {

            const auto &screen = coulomb_workspace.screening[iel];
            const auto &fspin = coulomb_workspace.fspin[iel];

            dcs::coulomb_data(
                    coulomb_workspace.fCM[iel],
                    screen, fspin,
                    coulomb_workspace.invlambda[iel],
                    table_K, element, MUON_MASS);

            dcs::coulomb_transport(
                    coulomb_workspace.G[iel],
                    screen, fspin,
                    torch::tensor(1.0, tensor_ops()));

            dcs::soft_scattering(
                    coulomb_workspace.table_ms1[iel],
                    table_K, element, MUON_MASS);
        }

        // TODO: implement CUDA version
        inline void compute_per_element_data() {
            const Index nel = num_elements();
            const auto &model_K = table_K.index(
                    {table_K >= dcs::DCS_MODEL_MIN_KINETIC});

            init_per_element_data(nel);
            init_dcs_data(nel, model_K.numel());

#pragma omp parallel for
            for (Index iel = 0; iel < nel; iel++) {
                const auto &element = get_element(iel);
                compute_recoil_energy_integrals(table_CSn[iel], dcs::del_integrand, element);
                compute_recoil_energy_integrals(cel_table[iel], dcs::cel_integrand, element);
                compute_dcs_model(dcs_data[iel], model_K, element);
                compute_coulomb_data(element, iel);
            }
        }

        inline Tabulation compute_be_cel(
                const Tabulation &br,
                const Tabulation &pp,
                const Tabulation &ph,
                const Tabulation &io,
                const Tabulation &cs) {
            auto be_cel = torch::zeros_like(br);
            be_cel += torch::where(cs[0] < br, cs[0], br);
            be_cel += torch::where(cs[1] < pp, cs[1], pp);
            be_cel += torch::where(cs[2] < ph, cs[2], ph);
            be_cel += torch::where(cs[3] < io, cs[3], io);
            return be_cel;
        }

        inline void compute_dedx_tables(
                MaterialId imat, mdf::DEDXTable &dedx_table) {

            const auto &material = get_material(imat);
            const Index nel = material.element_ids.numel();
            const Index nkin = table_K.numel();

            table_CSf[imat] = material.fractions.view({nel, 1, 1}) *
                              table_CSn.index_select(0, material.element_ids);
            table_CS[imat] = table_CSf[imat].sum(0).sum(0);

            table_CSf[imat] *= torch::where(table_CS[imat] <= 0.0,
                                            torch::tensor(0.0, tensor_ops()), torch::tensor(1.0, tensor_ops()))
                    .view({1, 1, nkin});
            table_CSf[imat] =
                    table_CSf[imat].view({nel * dcs::NPR, nkin}).cumsum(0).view({nel, dcs::NPR, nkin}) /
                    torch::where(table_CS[imat] <= 0.0, torch::tensor(1.0, tensor_ops()),
                                 table_CS[imat]).view({1, 1, nkin});

            const auto ionisation = dcs::DEDX_SCALE *
                                    torch::from_blob(dedx_table.Ionisation.data(), nkin, torch::kDouble).to(
                                            tensor_ops());

            auto be_cel = compute_be_cel(
                    dcs::DEDX_SCALE *
                    torch::from_blob(dedx_table.brems.data(), nkin, torch::kDouble).to(tensor_ops()),
                    dcs::DEDX_SCALE *
                    torch::from_blob(dedx_table.pair.data(), nkin, torch::kDouble).to(tensor_ops()),
                    dcs::DEDX_SCALE *
                    torch::from_blob(dedx_table.photonuc.data(), nkin, torch::kDouble).to(tensor_ops()),
                    ionisation,
                    torch::tensordot(
                            material.fractions,
                            cel_table.index_select(0, material.element_ids), 0, 0)
            );

            table_dE_CSDA[imat] = dcs::DEDX_SCALE *
                                  torch::from_blob(dedx_table.dEdX.data(), nkin, torch::kDouble).to(tensor_ops());
            table_dE[imat] = table_dE_CSDA[imat] - be_cel;
            table_NI_in[imat] = table_CS[imat] / table_dE[imat];
            table_a_max[imat] = ionisation[nkin - 1];
            table_b_max[imat] =
                    (dcs::DEDX_SCALE * dedx_table.Radloss[nkin - 1] - be_cel[nkin - 1]) /
                    (MUON_MASS + table_K[nkin - 1]);
        }

        inline void set_dedx_tables(mdf::MaterialsDEDXData &dedx_data) {
            const Index nmat = num_materials();
            init_dedx_tables(nmat);
            for (Index imat = 0; imat < nmat; imat++)
                compute_dedx_tables(
                        imat, std::get<mdf::DEDXTable>(dedx_data.at(get_material_name(imat)))
                );
            cel_table = Tabulation{}; // drop data in cel_table
        }

        inline ThresholdIndex compute_kinetic_threshold(const MaterialId &imat, const Index &nkin) {
            ThresholdIndex ri = 0;
            Scalar cs0 = 0.0;
            auto *cs = table_CS[imat].data_ptr<Scalar>();
            for (ri = 0; ri < nkin; ri++)
                if ((cs0 = cs[ri]) > 0)
                    break;
            auto *dE = table_dE[imat].data_ptr<Scalar>();
            auto *NI_in = table_NI_in[imat].data_ptr<Scalar>();
            for (ThresholdIndex i = 0; i < ri; i++) {
                cs[i] = cs0;
                NI_in[i] = cs0 / dE[i];
            }
            return ri;
        }

        inline void compute_del_thresholds(
                const Tabulation &result,
                const AtomicElement &element,
                const ThresholdIndex th_i) {
            dcs::compute_threshold(dcs::bremsstrahlung, result[0], table_K,
                                   dcs::X_FRACTION, element, MUON_MASS, th_i);
            dcs::compute_threshold(dcs::pair_production, result[1], table_K,
                                   dcs::X_FRACTION, element, MUON_MASS, th_i);
            dcs::compute_threshold(dcs::photonuclear, result[2], table_K,
                                   dcs::X_FRACTION, element, MUON_MASS, th_i);
            dcs::compute_threshold(dcs::ionisation, result[3], table_K,
                                   dcs::X_FRACTION, element, MUON_MASS, th_i);
        }

        // TODO: implement CUDA version
        inline void compute_del_thresholds() {
            ThresholdIndex th_i = 0;
            const Index nmat = num_materials();
            const Index nkin = table_K.numel();
            const Index nel = num_elements();

            for (Index imat = 0; imat < nmat; imat++) {
                ThresholdIndex ri = compute_kinetic_threshold(imat, nkin);
                table_Kt[imat] = table_K[ri];
                th_i = std::max(th_i, ri);
            }

            for (Index iel = 0; iel < nel; iel++)
                compute_del_thresholds(table_Xt[iel], get_element(iel), th_i);
        }

        // TODO: implement CUDA version
        inline void compute_coulomb_scattering_tables(const MaterialId imat) {
            const auto &elids = get_material(imat).element_ids;
            const auto &fracs = get_material(imat).fractions;
            const auto nel = elids.numel();

            const auto G = coulomb_workspace.G.index_select(0, elids);
            const auto fCM = coulomb_workspace.fCM.index_select(0, elids);
            const auto screen = coulomb_workspace.screening.index_select(0, elids);
            const auto invlambda =
                    coulomb_workspace.invlambda.index_select(0, elids) * fracs.view({nel, 1});
            const auto fspin = coulomb_workspace.fspin.index_select(0, elids);
            const auto ms1 = coulomb_workspace.table_ms1.index_select(0, elids) * fracs.view({nel, 1});

            const auto &mu0 = table_Mu0[imat];
            const auto lb_h = torch::zeros_like(mu0);

            dcs::hard_scattering(
                    mu0, lb_h, G, fCM, screen, invlambda, fspin);

            table_Lb[imat] = lb_h * table_K * (table_K + 2 * MUON_MASS);
            table_NI_el[imat] = 1. / (table_dE[imat] * lb_h);

            for (Index iel = 0; iel < nel; iel++) {
                const auto &Gi = G[iel];
                dcs::coulomb_transport(
                        Gi, screen[iel], fspin[iel], mu0);

                const auto &fCMi = fCM[iel];

                table_Ms1[imat] += ms1[iel] +
                                   invlambda[iel] * Gi.index({Ellipsis, 1}) *
                                   (1. / (fCMi.index({Ellipsis, 0}) * (1. + fCMi.index({Ellipsis, 1})))).pow(2);
            }
        }

        inline void set_coulomb_parameters() {
            const Index nmat = num_materials();
            init_coulomb_parameters();

            for (Index imat = 0; imat < nmat; imat++)
                compute_coulomb_scattering_tables(imat);

            coulomb_workspace = CoulombWorkspace{}; // drop CoulombWorkspace data
        }

        // TODO: implement CUDA version
        inline void set_cel_integrals() {
            const Index nmat = num_materials();
            init_cel_integrals(nmat);

            const Scalar I0 = dcs::compute_momentum_integral(table_K[0].item<Scalar>(), MUON_MASS);
            for (Index imat = 0; imat < nmat; imat++) {
                dcs::compute_cel_grammage_integral(table_X[imat], table_dE[imat], table_K);
                dcs::compute_cel_grammage_integral(table_X_CSDA[imat], table_dE_CSDA[imat], table_K);
                dcs::compute_time_integral(table_T[imat], table_X[imat], table_K, MUON_MASS, I0);
                dcs::compute_time_integral(table_T_CSDA[imat], table_X_CSDA[imat], table_K, MUON_MASS, I0);
                dcs::compute_kinetic_integral(table_NI_el[imat], table_K);
                dcs::compute_kinetic_integral(table_NI_in[imat], table_K);
                dcs::compute_csda_magnetic_transport(table_Li[imat], table_T_CSDA[imat], table_X_CSDA[imat],
                                                     MUON_MASS, LARMOR_FACTOR);
            }
        }

    public:

        inline MuonPhysics &set_mdf_settings(const mdf::Settings &mdf_settings) {
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings));
            return *this;
        }

        inline utils::Status load_dedx_data(mdf::MaterialsDEDXData &dedx_data) {
            if (!process_dedx_data_header(dedx_data))
                return false;
            if (!set_table_K(dedx_data))
                return false;

            compute_per_element_data();
            set_dedx_tables(dedx_data);
            compute_del_thresholds();
            set_coulomb_parameters();
            set_cel_integrals();

            return true;
        }

    };


} //namespace noa::pms::leptons
