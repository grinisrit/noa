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

#include "ghmc/pms/physics.hh"

#include <torch/torch.h>

namespace ghmc::pms
{
    using KineticEnergy = torch::Tensor;
    using RecoilEnergy = torch::Tensor;
    using DCSValues = torch::Tensor;

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9155
     */
    inline void default_dcs_bremsstrahlung(
        const AtomicNumber Z,
        const AtomicMass &A,
        const ParticleMass &mu,
        const float &K,
        const float &q,
        float &r)
    {
        const float me = ELECTRON_MASS;
        const float sqrte = 1.648721271f;
        const float phie_factor = mu / (me * me * sqrte);
        const float rem = 5.63588E-13f * me / mu;

        const float BZ_n = (Z == 1) ? 202.4f : 182.7f * pow(Z, -1.f / 3.f);
        const float BZ_e = (Z == 1) ? 446.f : 1429.f * pow(Z, -2.f / 3.f);
        const float D_n = 1.54f * pow(A, 0.27f);
        const float E = K + mu;
        const float dcs_factor = 7.297182E-07f * rem * rem * Z / E;

        const float delta_factor = 0.5f * mu * mu / E;
        const float qe_max = E / (1.f + 0.5f * mu * mu / (me * E));

        const float nu = q / E;
        const float delta = delta_factor * nu / (1.f - nu);
        float Phi_n, Phi_e;
        Phi_n = log(BZ_n * (mu + delta * (D_n * sqrte - 2.f)) /
                    (D_n * (me + delta * sqrte * BZ_n)));
        if (Phi_n < 0.f)
            Phi_n = 0.f;
        if (q < qe_max)
        {
            Phi_e = log(BZ_e * mu /
                        ((1.f + delta * phie_factor) * (me + delta * sqrte * BZ_e)));
            if (Phi_e < 0.f)
                Phi_e = 0.f;
        }
        else
            Phi_e = 0.f;

        const float dcs =
            dcs_factor * (Z * Phi_n + Phi_e) * (4.f / 3.f * (1.f / nu - 1.f) + nu);
        r = (dcs < 0.f) ? 0.f : dcs * 1E+03f * AVOGADRO_NUMBER * (mu + K) / A;
    }

    inline void dcs_bremsstrahlung(
        const AtomicNumber Z,
        const AtomicMass &A,
        const ParticleMass &mu,
        const KineticEnergy &K,
        const RecoilEnergy &q,
        DCSValues &res)
    {
        const float *pK = K.data_ptr<float>();
        const float *pq = q.data_ptr<float>();
        float *pres = res.data_ptr<float>();
        const int n = res.numel();
        for (int i = 0; i < n; i++)
            default_dcs_bremsstrahlung(Z, A, mu, pK[i], pq[i], pres[i]);
    }

} // namespace ghmc::pms
