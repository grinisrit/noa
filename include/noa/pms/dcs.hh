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

#include "noa/pms/constants.hh"
#include "noa/utils/common.hh"

#include <torch/types.h>

namespace noa::pms::dcs {

    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using Table = torch::Tensor;  // Generic table
    using Result = torch::Tensor; // Receiver tensor for calculations result

    template<typename Dtype, typename DCSFunc>
    inline auto vmap(const DCSFunc &dcs_func) {
        return [&dcs_func](const Result &result,
                           const KineticEnergies &kinetic_energies,
                           const RecoilEnergies &recoil_energies,
                           const AtomicElement <Dtype> &element,
                           const Dtype &mass) {
            const Dtype *recoil_energy = recoil_energies.data_ptr<Dtype>();
            utils::vmapi<Scalar>(
                    kinetic_energies,
                    [&](const int64_t i, const auto &k) { return dcs_func(k, recoil_energy[i], element, mass); },
                    result);
        };
    }

    template<typename Dtype, typename DCSFunc>
    inline auto pvmap(const DCSFunc &dcs_func) {
        return [&dcs_func](const Result &result,
                           const KineticEnergies &kinetic_energies,
                           const RecoilEnergies &recoil_energies,
                           const AtomicElement <Dtype> &element,
                           const Dtype &mass) {
            const Dtype *recoil_energy = recoil_energies.data_ptr<Dtype>();
            utils::pvmapi<Scalar>(
                    kinetic_energies,
                    [&](const int64_t i, const auto &k) { return dcs_func(k, recoil_energy[i], element, mass); },
                    result);
        };
    }

    namespace pumas {
        /*
        *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
        *  GNU Lesser General Public License version 3
        *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9155
        */
        inline const auto bremsstrahlung = [](
                const Energy kinetic_energy,
                const Energy recoil_energy,
                const AtomicElement <Scalar> element,
                const ParticleMass mass) {
            const int Z = element.Z;
            const Scalar A = element.A;
            const Scalar me = ELECTRON_MASS;
            const Scalar sqrte = 1.648721271;
            const Scalar phie_factor = mass / (me * me * sqrte);
            const Scalar rem = 5.63588E-13 * me / mass;

            const Scalar BZ_n = (Z == 1) ? 202.4 : 182.7 * pow(Z, -1. / 3.);
            const Scalar BZ_e = (Z == 1) ? 446. : 1429. * pow(Z, -2. / 3.);
            const Scalar D_n = 1.54 * pow(A, 0.27);
            const Scalar E = kinetic_energy + mass;
            const Scalar dcs_factor = 7.297182E-07 * rem * rem * Z / E;

            const Scalar delta_factor = 0.5 * mass * mass / E;
            const Scalar qe_max = E / (1. + 0.5 * mass * mass / (me * E));

            const Scalar nu = recoil_energy / E;
            const Scalar delta = delta_factor * nu / (1. - nu);
            Scalar Phi_n, Phi_e;
            Phi_n = log(BZ_n * (mass + delta * (D_n * sqrte - 2.)) /
                        (D_n * (me + delta * sqrte * BZ_n)));
            if (Phi_n < 0.)
                Phi_n = 0.;
            if (recoil_energy < qe_max) {
                Phi_e = log(BZ_e * mass /
                            ((1. + delta * phie_factor) * (me + delta * sqrte * BZ_e)));
                if (Phi_e < 0.)
                    Phi_e = 0.;
            } else
                Phi_e = 0.;

            const Scalar dcs =
                    dcs_factor * (Z * Phi_n + Phi_e) * (4. / 3. * (1. / nu - 1.) + nu);
            return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mass + kinetic_energy) / A;
        };
    } // namespace noa::pms::dcs::pumas
} // namespace noa::pms::dcs