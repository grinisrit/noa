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

#include "noa/pms/pms.hh"
#include "noa/pms/mdf.hh"
#include "noa/pms/constants.hh"

#include <torch/torch.h>

namespace noa::pms {

    using MaterialsRelativeElectronicDensity = torch::Tensor;
    using MaterialsMeanExcitationEnergy = torch::Tensor;
    using MaterialsDensityEffect = std::vector<MaterialDensityEffect < Scalar>>;
    
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

    struct CoulombWorkspace {
        pumas::TransportCoefs G;
        pumas::CMLorentz fCM;
        pumas::ScreeningFactors screening;
        pumas::InvLambdas invlambda;
        pumas::FSpins fspin;
        pumas::SoftScatter table_ms1;
    };

    
    class MuonPhysics : public Model<Scalar, MuonPhysics> {

        friend class Model<Scalar, MuonPhysics>;

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
        CoulombWorkspaceRef coulomb_workspace;

        // TODO: parametrise MuonPhysics by device when CUDA support available
        inline c10::TensorOptions tensor_ops() const {
            return torch::dtype(c10::CppTypeToScalarType<Scalar>{}).layout(torch::kStrided);
        }

        inline AtomicElement <Scalar> process_element(const AtomicElement <Scalar> &element) const {
            return AtomicElement<Scalar>{element.A, 1E-6 * pumas::ENERGY_SCALE * element.I, element.Z};
        }

        inline Material <Scalar> process_material(
                const Material <Scalar> &material) const {
            return Material<Scalar>{
                    material.element_ids,
                    material.fractions.to(tensor_ops()),
                    pumas::DENSITY_SCALE * material.density};
        }

        inline utils::Status process_dedx_data_header(mdf::MaterialsDEDXData &dedx_data) {
            if (!mdf::check_particle_mass(MUON_MASS / pumas::ENERGY_SCALE, dedx_data))
                return false;

            const int nmat = num_materials();
            material_ZoA = torch::zeros(nmat, tensor_ops());
            material_I = torch::zeros(nmat, tensor_ops());

            for (int i = 0; i < nmat; i++) {
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
            const int nkin = values.size();
            auto tensor = torch::from_blob(values.data(), nkin, torch::kDouble);
            table_K = tensor.to(tensor_ops()) * pumas::ENERGY_SCALE;
            data++;
            for (auto &it = data; it != dedx_data.end(); it++) {
                auto &it_vals = std::get<mdf::DEDXTable>(it->second).T;
                auto it_ten = torch::from_blob(
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

        inline void init_dcs_data(const int nel, const int nkin) {
            dcs_data = torch::zeros({nel, pumas::NPR - 1, nkin, pumas::NDM}, tensor_ops());
        }

        inline void init_per_element_data(const int nel) {
            const int nkin = table_K.numel();
            table_CSn = torch::zeros({nel, pumas::NPR, nkin}, tensor_ops());
            cel_table = torch::zeros_like(table_CSn);
            coulomb_workspace = CoulombWorkspaceRef{
                    torch::zeros({nel, nkin, 2}, tensor_ops()),
                    torch::zeros({nel, nkin, 2}, tensor_ops()),
                    torch::zeros({nel, nkin, dcs::NSF}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops()),
                    torch::zeros({nel, nkin}, tensor_ops())};
        }

        inline void compute_per_element_data() {
            const int nel = num_elements();
            const auto &model_K = table_K.index(
                    {table_K >= pumas::DCS_MODEL_MIN_KINETIC});

            init_per_element_data(nel);
            init_dcs_data(nel, model_K.numel());
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
            return true;
        }

    };


} //namespace noa::pms
