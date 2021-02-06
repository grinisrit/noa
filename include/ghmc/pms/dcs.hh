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
    using KineticEnergy = float;
    using RecoilEnergy = float;
    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;

    template <typename DCS>
    inline auto map_dcs(const DCS &dcs)
    {
        return [&dcs](const AtomicNumber Z,
                      const AtomicMass &A,
                      const ParticleMass &mu,
                      const KineticEnergies &K,
                      const RecoilEnergies &q) {
            const float *pK = K.data_ptr<float>();
            const float *pq = q.data_ptr<float>();
            auto res = torch::zeros_like(K);
            float *pres = res.data_ptr<float>();
            const int n = res.numel();
            for (int i = 0; i < n; i++)
                pres[i] = dcs(Z, A, mu, pK[i], pq[i]);
            return res;
        };
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9155
     */
    inline const auto default_dcs_bremsstrahlung = [](
                                                       const AtomicNumber Z,
                                                       const AtomicMass &A,
                                                       const ParticleMass &mu,
                                                       const KineticEnergy &K,
                                                       const RecoilEnergy &q) {
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
        return (dcs < 0.f) ? 0.f : dcs * 1E+03f * AVOGADRO_NUMBER * (mu + K) / A;
    };

    inline const auto dcs_bremsstrahlung = map_dcs(default_dcs_bremsstrahlung);

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
     */
    inline const auto default_dcs_pair_production = [](
                                                        const AtomicNumber Z,
                                                        const AtomicMass &A_,
                                                        const ParticleMass &mu,
                                                        const KineticEnergy &K,
                                                        const RecoilEnergy &q) {
        constexpr int N_GQ = 8;
        const float xGQ[N_GQ] = {0.01985507f, 0.10166676f, 0.2372338f,
                                 0.40828268f, 0.59171732f, 0.7627662f, 0.89833324f, 0.98014493f};
        const float wGQ[N_GQ] = {0.05061427f, 0.11119052f, 0.15685332f,
                                 0.18134189f, 0.18134189f, 0.15685332f, 0.11119052f, 0.05061427f};

        //  Check the bounds of the energy transfer.
        if (q <= 4.f * ELECTRON_MASS)
            return 0.f;

        const float sqrte = 1.6487212707f;
        const float Z13 = pow(Z, 1.f / 3.f);
        if (q >= K + mu * (1.f - 0.75f * sqrte * Z13))
            return 0.f;

        //  Precompute some constant factors for the integration.
        const float nu = q / (K + mu);
        const float r = mu / ELECTRON_MASS;
        const float beta = 0.5f * nu * nu / (1.f - nu);
        const float xi_factor = 0.5f * r * r * beta;
        const float A = (Z == 1) ? 202.4f : 183.f;
        const float AZ13 = A / Z13;
        const float cL = 2.f * sqrte * ELECTRON_MASS * AZ13;
        const float cLe = 2.25f * Z13 * Z13 / (r * r);

        //  Compute the bound for the integral.
        const float gamma = 1.f + K / mu;
        const float x0 = 4.f * ELECTRON_MASS / q;
        const float x1 = 6.f / (gamma * (gamma - q / mu));
        const float argmin =
            (x0 + 2.f * (1.f - x0) * x1) / (1.f + (1.f - x1) * sqrt(1.f - x0));
        if ((argmin >= 1.f) || (argmin <= 0.f))
            return 0.f;
        const float tmin = log(argmin);

        //  Compute the integral over t = ln(1-rho).
        float I = 0.f;
        int i;
        for (i = 0; i < N_GQ; i++)
        {
            const float eps = exp(xGQ[i] * tmin);
            const float rho = 1.f - eps;
            const float rho2 = rho * rho;
            const float rho21 = eps * (2.f - eps);
            const float xi = xi_factor * rho21;
            const float xi_i = 1.f / xi;

            // Compute the e-term.
            float Be;
            if (xi >= 1E+03f)
                Be =
                    0.5f * xi_i * ((3.f - rho2) + 2.f * beta * (1.f + rho2));
            else
                Be = ((2.f + rho2) * (1.f + beta) + xi * (3.f + rho2)) *
                         log(1.f + xi_i) +
                     (rho21 - beta) / (1.f + xi) - 3.f - rho2;
            const float Ye = (5.f - rho2 + 4.f * beta * (1.f + rho2)) /
                             (2.f * (1.f + 3.f * beta) * log(3.f + xi_i) - rho2 -
                              2.f * beta * (2.f - rho2));
            const float xe = (1.f + xi) * (1.f + Ye);
            const float cLi = cL / rho21;
            const float Le = log(AZ13 * sqrt(xe) * q / (q + cLi * xe)) -
                             0.5f * log(1.f + cLe * xe);
            float Phi_e = Be * Le;
            if (Phi_e < 0.f)
                Phi_e = 0.f;

            // Compute the mu-term.
            float Bmu;
            if (xi <= 1E-03f)
                Bmu = 0.5f * xi * (5.f - rho2 + beta * (3.f + rho2));
            else
                Bmu = ((1.f + rho2) * (1.f + 1.5f * beta) -
                       xi_i * (1.f + 2.f * beta) * rho21) *
                          log(1.f + xi) +
                      xi * (rho21 - beta) / (1. + xi) +
                      (1.f + 2.f * beta) * rho21;
            const float Ymu = (4.f + rho2 + 3.f * beta * (1.f + rho2)) /
                              ((1.f + rho2) * (1.5f + 2.f * beta) * log(3.f + xi) + 1.f -
                               1.5f * rho2);
            const float xmu = (1.f + xi) * (1.f + Ymu);
            const float Lmu =
                log(r * AZ13 * q / (1.5f * Z13 * (q + cLi * xmu)));
            float Phi_mu = Bmu * Lmu;
            if (Phi_mu < 0.f)
                Phi_mu = 0.f;

            // Update the t-integral.
            I -= (Phi_e + Phi_mu / (r * r)) * (1.f - rho) * wGQ[i] * tmin;
        }

        // Atomic electrons form factor.
        float zeta;
        if (gamma <= 35.f)
            zeta = 0.f;
        else
        {
            float gamma1, gamma2;
            if (Z == 1)
            {
                gamma1 = 4.4E-05f;
                gamma2 = 4.8E-05f;
            }
            else
            {
                gamma1 = 1.95E-05f;
                gamma2 = 5.30E-05f;
            }
            zeta = 0.073f * log(gamma / (1.f + gamma1 * gamma * Z13 * Z13)) -
                   0.26f;
            if (zeta <= 0.f)
                zeta = 0.f;
            else
            {
                zeta /=
                    0.058f * log(gamma / (1.f + gamma2 * gamma * Z13)) -
                    0.14f;
            }
        }

        // Gather the results and return the macroscopic DCS.
        const float E = K + mu;
        const float dcs = 1.794664E-34f * Z * (Z + zeta) * (E - q) * I /
                          (q * E);
        return (dcs < 0.f) ? 0.f : dcs * 1E+03f * AVOGADRO_NUMBER * (mu + K) / A_;
    };

    inline const auto dcs_pair_production = map_dcs(default_dcs_pair_production);

} // namespace ghmc::pms
