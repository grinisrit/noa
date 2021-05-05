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

#include <torch/torch.h>

namespace noa::pms {

    using MaterialsRelativeElectronicDensity = torch::Tensor;
    using MaterialsMeanExcitationEnergy = torch::Tensor;
    using MaterialsDensityEffect = std::vector<MaterialDensityEffect<Scalar>>;

    constexpr Scalar ENERGY_SCALE = 1E-3; // from MeV to GeV
    constexpr Scalar DENSITY_SCALE = 1E+3; // from g/cm^3 to kg/m^3

    // TODO: parametrise MuonPhysics by device when CUDA support available
    inline const auto tensor_ops = torch::dtype(c10::CppTypeToScalarType<Scalar>{}).layout(torch::kStrided);

    class MuonPhysics : public Model<Scalar, MuonPhysics> {

        friend class Model<Scalar, MuonPhysics>;

        MaterialsRelativeElectronicDensity material_ZoA;
        MaterialsMeanExcitationEnergy material_I;
        MaterialsDensityEffect material_density_effect;

        inline AtomicElement <Scalar> process_element(const AtomicElement <Scalar> &element) const {
            return AtomicElement<Scalar>{element.A, 1E-6 * ENERGY_SCALE * element.I, element.Z};
        }

        inline Material <Scalar> process_material(
                const Material <Scalar> &material) const {
            return Material<Scalar>{
                material.element_ids,
                material.fractions.to(tensor_ops),
                DENSITY_SCALE * material.density};
        }

        inline utils::Status process_dedx_data_header(mdf::MaterialsDEDXData &dedx_data)
        {
            if(!mdf::check_particle_mass(MUON_MASS / ENERGY_SCALE, dedx_data))
                return false;

            const int nmat = num_materials();
            material_ZoA = torch::zeros(nmat, tensor_ops);
            material_I = torch::zeros(nmat, tensor_ops);

            for(int i = 0; i< nmat; i++){
                const auto &coefs = std::get<mdf::DEDXMaterialCoefficients>(
                        dedx_data.at(get_material_name(i)));
                material_ZoA[i] = coefs.ZoA;
                material_I[i] = coefs.I;
                material_density_effect.push_back(coefs.density_effect);
            }
            return true;
        }

    public:

        inline MuonPhysics &set_mdf_settings(const mdf::Settings &mdf_settings){
            set_elements(std::get<mdf::Elements>(mdf_settings));
            set_materials(std::get<mdf::Materials>(mdf_settings));
            return *this;
        }

        inline utils::Status load_dedx_data(mdf::MaterialsDEDXData &dedx_data) {
            if(!process_dedx_data_header(dedx_data))
                return false;
            return true;
        }


    };


} //namespace noa::pms
