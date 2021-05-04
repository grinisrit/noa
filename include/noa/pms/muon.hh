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

namespace noa::pms {

    using MaterialRelativeElectronicDensities = torch::Tensor;
    using MaterialMeanExcitationEnergies = torch::Tensor;

    constexpr Scalar ENERGY_SCALE = 1E-3; // from MeV to GeV
    constexpr Scalar DENSITY_SCALE = 1E+3; // from g/cm^3 to kg/m^3

    inline const auto tensor_ops = torch::dtype(torch::kDouble).layout(torch::kStrided);

    class MuonPhysics : public Model<Scalar, MuonPhysics> {

        friend class Model<Scalar, MuonPhysics>;

        inline AtomicElement <Scalar> process_element(const AtomicElement <Scalar> &element) const {
            return AtomicElement<Scalar>{element.A, 1E-6 * ENERGY_SCALE * element.I, element.Z};
        }

        inline const Material <Scalar> process_material(
                const Material <Scalar> &material) const {
            return Material<Scalar>{
                material.element_ids,
                material.fractions.to(tensor_ops),
                DENSITY_SCALE * material.density};
        }


    };


} //namespace noa::pms
