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

    template <typename DCSKernel>
    inline auto map_dcs_kernel(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const AtomicNumber Z,
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
                pres[i] = dcs_kernel(Z, A, mu, pK[i], pq[i]);
            return res;
        };
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9155
     */
    inline const auto dcs_bremsstrahlung_kernel = [](
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

    inline const auto dcs_bremsstrahlung = map_dcs_kernel(dcs_bremsstrahlung_kernel);

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
     */
    inline const auto dcs_pair_production_kernel = [](
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

    inline const auto dcs_pair_production = map_dcs_kernel(dcs_pair_production_kernel);

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9371
     */
    inline float dcs_photonuclear_f2_allm(const float &x, const float &Q2)
    {
        const float m02 = 0.31985f;
        const float mP2 = 49.457f;
        const float mR2 = 0.15052f;
        const float Q02 = 0.52544f;
        const float Lambda2 = 0.06527f;

        const float cP1 = 0.28067f;
        const float cP2 = 0.22291f;
        const float cP3 = 2.1979f;
        const float aP1 = -0.0808f;
        const float aP2 = -0.44812f;
        const float aP3 = 1.1709f;
        const float bP1 = 0.36292f;
        const float bP2 = 1.8917f;
        const float bP3 = 1.8439f;

        const float cR1 = 0.80107f;
        const float cR2 = 0.97307f;
        const float cR3 = 3.4942f;
        const float aR1 = 0.58400f;
        const float aR2 = 0.37888f;
        const float aR3 = 2.6063f;
        const float bR1 = 0.01147f;
        const float bR2 = 3.7582f;
        const float bR3 = 0.49338f;

        const float M2 = 0.8803505929f;
        const float W2 = M2 + Q2 * (1.f / x - 1.f);
        const float t = log(log((Q2 + Q02) / Lambda2) / log(Q02 / Lambda2));
        const float xP = (Q2 + mP2) / (Q2 + mP2 + W2 - M2);
        const float xR = (Q2 + mR2) / (Q2 + mR2 + W2 - M2);
        const float lnt = log(t);
        const float cP =
            cP1 + (cP1 - cP2) * (1.f / (1.f + exp(cP3 * lnt)) - 1.f);
        const float aP =
            aP1 + (aP1 - aP2) * (1.f / (1.f + exp(aP3 * lnt)) - 1.f);
        const float bP = bP1 + bP2 * exp(bP3 * lnt);
        const float cR = cR1 + cR2 * exp(cR3 * lnt);
        const float aR = aR1 + aR2 * exp(aR3 * lnt);
        const float bR = bR1 + bR2 * exp(bR3 * lnt);

        const float F2P = cP * exp(aP * log(xP) + bP * log(1 - x));
        const float F2R = cR * exp(aR * log(xR) + bR * log(1 - x));

        return Q2 / (Q2 + m02) * (F2P + F2R);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9433
     */
    inline float dcs_photonuclear_f2a_drss(const float &x, const float &F2p, const float &A)
    {
        float a = 1.f;
        if (x < 0.0014f)
            a = exp(-0.1f * log(A));
        else if (x < 0.04f)
            a = exp((0.069f * log10(x) + 0.097f) * log(A));

        return (0.5f * A * a *
                (2.f + x * (-1.85f + x * (2.45f + x * (-2.35f + x)))) * F2p);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9453
     */
    inline float dcs_photonuclear_r_whitlow(const float &x, const float &Q2)
    {
        float q2 = Q2;
        if (Q2 < 0.3f)
            q2 = 0.3f;

        const float theta =
            1.f + 12.f * q2 / (1.f + q2) * 0.015625f / (0.015625f + x * x);

        return (0.635f / log(q2 / 0.04f) * theta + 0.5747f / q2 -
                0.3534f / (0.09f + q2 * q2));
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9478
     */

    float dcs_photonuclear_d2(const float &A, const float &ml, const float &K, const float &q, const float &Q2)
    {
        const float cf = 2.603096E-35f;
        const float M = 0.931494f;
        const float E = K + ml;

        const float y = q / E;
        const float x = 0.5f * Q2 / (M * q);
        const float F2p = dcs_photonuclear_f2_allm(x, Q2);
        const float F2A = dcs_photonuclear_f2a_drss(x, F2p, A);
        const float R = dcs_photonuclear_r_whitlow(x, Q2);

        const float dds = (1.f - y +
                           0.5f * (1.f - 2.f * ml * ml / Q2) *
                               (y * y + Q2 / (E * E)) / (1.f + R)) /
                              (Q2 * Q2) -
                          0.25f / (E * E * Q2);

        return cf * F2A * dds / q;
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9515
     */
    inline const auto dcs_photonuclear_kernel = [](
                                                    const AtomicNumber,
                                                    const AtomicMass &A,
                                                    const ParticleMass &mu,
                                                    const KineticEnergy &K,
                                                    const RecoilEnergy &q) {
        constexpr int N_GQ = 9;
        const float xGQ[N_GQ] = {0.0000000000000000f, -0.8360311073266358f,
                                 0.8360311073266358f, -0.9681602395076261f, 0.9681602395076261f,
                                 -0.3242534234038089f, 0.3242534234038089f, -0.6133714327005904f,
                                 0.6133714327005904f};
        const float wGQ[N_GQ] = {0.3302393550012598f, 0.1806481606948574f,
                                 0.1806481606948574f, 0.0812743883615744f, 0.0812743883615744f,
                                 0.3123470770400029f, 0.3123470770400029f, 0.2606106964029354f,
                                 0.2606106964029354f};

        const float M = 0.931494f;
        const float mpi = 0.134977f;
        const float E = K + mu;

        float ds = 0.f;
        if ((q >= (E - mu)) || (q <= (mpi * (1.f + 0.5f * mpi / M))))
            return ds;

        const float y = q / E;
        const float Q2min = mu * mu * y * y / (1.f - y);
        const float Q2max = 2.f * M * (q - mpi) - mpi * mpi;
        if ((Q2max < Q2min) | (Q2min < 0.f))
            return ds;

        /* Set the binning. */
        const float pQ2min = log(Q2min);
        const float pQ2max = log(Q2max);
        const float dpQ2 = pQ2max - pQ2min;
        const float pQ2c = 0.5f * (pQ2max + pQ2min);

        /*
         * Integrate the doubly differential cross-section over Q2 using
         * a Gaussian quadrature. Note that 9 points are enough to get a
         * better than 0.1 % accuracy.
         */
        int i;
        for (i = 0; i < N_GQ; i++)
        {
            const float Q2 = exp(pQ2c + 0.5f * dpQ2 * xGQ[i]);
            ds += dcs_photonuclear_d2(A, mu, K, q, Q2) * Q2 * wGQ[i];
        }

        return (ds < 0.f) ? 0.f : 0.5f * ds * dpQ2 * 1E+03f * AVOGADRO_NUMBER * (mu + K) / A;
    };

    inline const auto dcs_photonuclear = map_dcs_kernel(dcs_photonuclear_kernel);

} // namespace ghmc::pms
