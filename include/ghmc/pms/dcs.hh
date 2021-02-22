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

#include "ghmc/pms/physics.hh"
#include "ghmc/utils/common.hh"
#include "ghmc/utils/numerics.hh"

#include <torch/torch.h>

namespace ghmc::pms::dcs
{
    constexpr int NPR = 4; // Number of DEL processes considered
    constexpr int NSF = 9; // Number of screening factors and pole reduction for Coulomb scattering

    using KineticEnergy = Scalar;
    using RecoilEnergy = Scalar;
    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using Table = torch::Tensor;  // Generic table
    using Result = torch::Tensor; // Receiver tensor for calculations result
    using ComputeCEL = bool;      // Compute Continuous Energy Loss (CEL) flag

    using ThresholdIndex = Index;
    using Thresholds = torch::Tensor;

    using InvLambdas = torch::Tensor;       // Inverse of the mean free grammage
    using ScreeningFactors = torch::Tensor; // Atomic and nuclear screening factors & pole reduction
    using FSpins = torch::Tensor;           // Spin corrections
    using CMLorentz = torch::Tensor;        // Center of Mass to Observer frame transorm
    using AngularCutoff = torch::Tensor;    // Cutoff angle for coulomb scattering
    using HSMeanFreePath = torch::Tensor;   // Hard scattering mean free path
    using TransportCoefs = torch::Tensor;
    using SoftScatter = torch::Tensor; // Soft scattering terms per element

    template <typename DCSKernel>
    inline auto map_kernel(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const Result &result,
                             const KineticEnergies &K,
                             const RecoilEnergies &q,
                             const AtomicElement &element,
                             const ParticleMass &mass) {
            const double *pq = q.data_ptr<double>();
            utils::vmapi<double>(
                K,
                [&](const int i, const auto &k) { return dcs_kernel(k, pq[i], element, mass); },
                result);
        };
    }

    template <typename DCSKernel>
    inline auto compute_integral(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const KineticEnergy &K,
                             const EnergyTransferMin &xlow,
                             const AtomicElement &element,
                             const ParticleMass &mass,
                             const int min_points,
                             const ComputeCEL cel = false) {
            return utils::numerics::quadrature6<double>(
                       log(K * xlow), log(K),
                       [&](const double &t) {
                           const double q = exp(t);
                           double s = dcs_kernel(K, q, element, mass) * q;
                           if (cel)
                               s *= q;
                           return s;
                       },
                       min_points) /
                   (K + mass);
        };
    }

    template <typename DCSKernel>
    inline auto map_compute_integral(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const Result &result,
                             const KineticEnergies &K,
                             const EnergyTransferMin &xlow,
                             const AtomicElement &element,
                             const ParticleMass &mass,
                             const int min_points,
                             const ComputeCEL cel = false) {
            utils::vmap<double>(
                K,
                [&](const double &k) { return compute_integral(dcs_kernel)(
                                           k, xlow, element, mass, min_points, cel); },
                result);
        };
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9155
     */
    inline const auto default_bremsstrahlung = [](const KineticEnergy &K,
                                                  const RecoilEnergy &q,
                                                  const AtomicElement &element,
                                                  const ParticleMass &mass) {
        const int Z = element.Z;
        const auto A = element.A;
        const double me = ELECTRON_MASS;
        const double sqrte = 1.648721271;
        const double phie_factor = mass / (me * me * sqrte);
        const double rem = 5.63588E-13 * me / mass;

        const double BZ_n = (Z == 1) ? 202.4 : 182.7 * pow(Z, -1. / 3.);
        const double BZ_e = (Z == 1) ? 446. : 1429. * pow(Z, -2. / 3.);
        const double D_n = 1.54 * pow(A, 0.27);
        const double E = K + mass;
        const double dcs_factor = 7.297182E-07 * rem * rem * Z / E;

        const double delta_factor = 0.5 * mass * mass / E;
        const double qe_max = E / (1. + 0.5 * mass * mass / (me * E));

        const double nu = q / E;
        const double delta = delta_factor * nu / (1. - nu);
        double Phi_n, Phi_e;
        Phi_n = log(BZ_n * (mass + delta * (D_n * sqrte - 2.)) /
                    (D_n * (me + delta * sqrte * BZ_n)));
        if (Phi_n < 0.)
            Phi_n = 0.;
        if (q < qe_max)
        {
            Phi_e = log(BZ_e * mass /
                        ((1. + delta * phie_factor) * (me + delta * sqrte * BZ_e)));
            if (Phi_e < 0.)
                Phi_e = 0.;
        }
        else
            Phi_e = 0.;

        const double dcs =
            dcs_factor * (Z * Phi_n + Phi_e) * (4. / 3. * (1. / nu - 1.) + nu);
        return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mass + K) / A;
    };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
     */
    inline const auto default_pair_production = [](const KineticEnergy &K,
                                                   const RecoilEnergy &q,
                                                   const AtomicElement &element,
                                                   const ParticleMass &mass) {
        const int Z = element.Z;
        const double A = element.A;
        /*  Check the bounds of the energy transfer. */
        if (q <= 4. * ELECTRON_MASS)
            return 0.;
        const double sqrte = 1.6487212707;
        const double Z13 = pow(Z, 1. / 3.);
        if (q >= K + mass * (1. - 0.75 * sqrte * Z13))
            return 0.;

        /*  Precompute some constant factors for the compute_integral. */
        const double nu = q / (K + mass);
        const double r = mass / ELECTRON_MASS;
        const double beta = 0.5 * nu * nu / (1. - nu);
        const double xi_factor = 0.5 * r * r * beta;
        const double A_ = (Z == 1) ? 202.4 : 183.;
        const double AZ13 = A_ / Z13;
        const double cL = 2. * sqrte * ELECTRON_MASS * AZ13;
        const double cLe = 2.25 * Z13 * Z13 / (r * r);

        /*  Compute the bound for the integral. */
        const double gamma = 1. + K / mass;
        const double x0 = 4. * ELECTRON_MASS / q;
        const double x1 = 6. / (gamma * (gamma - q / mass));
        const double argmin =
            (x0 + 2. * (1. - x0) * x1) / (1. + (1. - x1) * sqrt(1. - x0));
        if ((argmin >= 1.) || (argmin <= 0.))
            return 0.;
        const double tmin = log(argmin);

        /*  Compute the integral over t = ln(1-rho). */
        double I = utils::numerics::quadrature8<double>(0.f, 1.f, [&](const double &t) {
            const double eps = exp(t * tmin);
            const double rho = 1. - eps;
            const double rho2 = rho * rho;
            const double rho21 = eps * (2. - eps);
            const double xi = xi_factor * rho21;
            const double xi_i = 1. / xi;

            /* Compute the e-term. */
            double Be;
            if (xi >= 1E+03)
                Be =
                    0.5 * xi_i * ((3 - rho2) + 2. * beta * (1. + rho2));
            else
                Be = ((2. + rho2) * (1. + beta) + xi * (3. + rho2)) *
                         log(1. + xi_i) +
                     (rho21 - beta) / (1. + xi) - 3. - rho2;
            const double Ye = (5. - rho2 + 4. * beta * (1. + rho2)) /
                              (2. * (1. + 3. * beta) * log(3. + xi_i) - rho2 -
                               2. * beta * (2. - rho2));
            const double xe = (1. + xi) * (1. + Ye);
            const double cLi = cL / rho21;
            const double Le = log(AZ13 * sqrt(xe) * q / (q + cLi * xe)) -
                              0.5 * log(1. + cLe * xe);
            double Phi_e = Be * Le;
            if (Phi_e < 0.)
                Phi_e = 0.;

            /* Compute the mass-term. */
            double Bmu;
            if (xi <= 1E-03)
                Bmu = 0.5 * xi * (5. - rho2 + beta * (3. + rho2));
            else
                Bmu = ((1. + rho2) * (1. + 1.5 * beta) -
                       xi_i * (1. + 2. * beta) * rho21) *
                          log(1. + xi) +
                      xi * (rho21 - beta) / (1. + xi) +
                      (1. + 2. * beta) * rho21;
            const double Ymu = (4. + rho2 + 3. * beta * (1. + rho2)) /
                               ((1. + rho2) * (1.5 + 2. * beta) * log(3. + xi) + 1. -
                                1.5 * rho2);
            const double xmu = (1. + xi) * (1. + Ymu);
            const double Lmu =
                log(r * AZ13 * q / (1.5 * Z13 * (q + cLi * xmu)));
            double Phi_mu = Bmu * Lmu;
            if (Phi_mu < 0.)
                Phi_mu = 0.;
            return -(Phi_e + Phi_mu / (r * r)) * (1. - rho) * tmin;
        });

        /* Atomic electrons form factor. */
        double zeta;
        if (gamma <= 35.)
            zeta = 0.;
        else
        {
            double gamma1, gamma2;
            if (Z == 1.)
            {
                gamma1 = 4.4E-05;
                gamma2 = 4.8E-05;
            }
            else
            {
                gamma1 = 1.95E-05;
                gamma2 = 5.30E-05;
            }
            zeta = 0.073 * log(gamma / (1. + gamma1 * gamma * Z13 * Z13)) -
                   0.26;
            if (zeta <= 0.)
                zeta = 0.;
            else
            {
                zeta /=
                    0.058 * log(gamma / (1. + gamma2 * gamma * Z13)) -
                    0.14;
            }
        }

        /* Gather the results and return the macroscopic DCS. */
        const double E = K + mass;
        const double dcs = 1.794664E-34 * Z * (Z + zeta) * (E - q) * I /
                           (q * E);
        return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mass + K) / A;
    };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9371
     */

    inline double dcs_photonuclear_f2_allm(const double x, const double Q2)
    {
        const double m02 = 0.31985;
        const double mP2 = 49.457;
        const double mR2 = 0.15052;
        const double Q02 = 0.52544;
        const double Lambda2 = 0.06527;

        const double cP1 = 0.28067;
        const double cP2 = 0.22291;
        const double cP3 = 2.1979;
        const double aP1 = -0.0808;
        const double aP2 = -0.44812;
        const double aP3 = 1.1709;
        const double bP1 = 0.36292;
        const double bP2 = 1.8917;
        const double bP3 = 1.8439;

        const double cR1 = 0.80107;
        const double cR2 = 0.97307;
        const double cR3 = 3.4942;
        const double aR1 = 0.58400;
        const double aR2 = 0.37888;
        const double aR3 = 2.6063;
        const double bR1 = 0.01147;
        const double bR2 = 3.7582;
        const double bR3 = 0.49338;

        const double M2 = 0.8803505929;
        const double W2 = M2 + Q2 * (1.0 / x - 1.0);
        const double t = log(log((Q2 + Q02) / Lambda2) / log(Q02 / Lambda2));
        const double xP = (Q2 + mP2) / (Q2 + mP2 + W2 - M2);
        const double xR = (Q2 + mR2) / (Q2 + mR2 + W2 - M2);
        const double lnt = log(t);
        const double cP =
            cP1 + (cP1 - cP2) * (1.0 / (1.0 + exp(cP3 * lnt)) - 1.0);
        const double aP =
            aP1 + (aP1 - aP2) * (1.0 / (1.0 + exp(aP3 * lnt)) - 1.0);
        const double bP = bP1 + bP2 * exp(bP3 * lnt);
        const double cR = cR1 + cR2 * exp(cR3 * lnt);
        const double aR = aR1 + aR2 * exp(aR3 * lnt);
        const double bR = bR1 + bR2 * exp(bR3 * lnt);

        const double F2P = cP * exp(aP * log(xP) + bP * log(1 - x));
        const double F2R = cR * exp(aR * log(xR) + bR * log(1 - x));

        return Q2 / (Q2 + m02) * (F2P + F2R);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9433
     */
    inline double dcs_photonuclear_f2a_drss(const double x, const double F2p, const double A)
    {
        double a = 1.0;
        if (x < 0.0014)
            a = exp(-0.1 * log(A));
        else if (x < 0.04)
            a = exp((0.069 * log10(x) + 0.097) * log(A));

        return (0.5 * A * a *
                (2.0 + x * (-1.85 + x * (2.45 + x * (-2.35 + x)))) * F2p);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9453
     */
    inline double dcs_photonuclear_r_whitlow(const double x, const double Q2)
    {
        double q2 = Q2;
        if (Q2 < 0.3)
            q2 = 0.3;

        const double theta =
            1 + 12.0 * q2 / (1.0 + q2) * 0.015625 / (0.015625 + x * x);

        return (0.635 / log(q2 / 0.04) * theta + 0.5747 / q2 -
                0.3534 / (0.09 + q2 * q2));
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9478
     */

    inline double dcs_photonuclear_d2(const double A, const double mass, const double K, const double q, const double Q2)
    {
        const double cf = 2.603096E-35;
        const double M = 0.931494;
        const double E = K + mass;

        const double y = q / E;
        const double x = 0.5 * Q2 / (M * q);
        const double F2p = dcs_photonuclear_f2_allm(x, Q2);
        const double F2A = dcs_photonuclear_f2a_drss(x, F2p, A);
        const double R = dcs_photonuclear_r_whitlow(x, Q2);

        const double dds = (1 - y +
                            0.5 * (1 - 2 * mass * mass / Q2) *
                                (y * y + Q2 / (E * E)) / (1 + R)) /
                               (Q2 * Q2) -
                           0.25 / (E * E * Q2);

        return cf * F2A * dds / q;
    }

    inline bool dcs_photonuclear_check(const double K, const double q)
    {
        return (q < 1.) || (q < 2E-03 * K);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9515
     */
    inline const auto default_photonuclear = [](const KineticEnergy &K,
                                                const RecoilEnergy &q,
                                                const AtomicElement &element,
                                                const ParticleMass &mass) {
        if (dcs_photonuclear_check(K, q))
            return 0.;

        const double A = element.A;
        const double M = 0.931494;
        const double mpi = 0.134977;
        const double E = K + mass;

        if ((q >= (E - mass)) || (q <= (mpi * (1.0 + 0.5 * mpi / M))))
            return 0.;

        const double y = q / E;
        const double Q2min = mass * mass * y * y / (1 - y);
        const double Q2max = 2.0 * M * (q - mpi) - mpi * mpi;
        if ((Q2max < Q2min) | (Q2min < 0))
            return 0.;

        /* Set the binning. */
        const double pQ2min = log(Q2min);
        const double pQ2max = log(Q2max);
        const double dpQ2 = pQ2max - pQ2min;
        const double pQ2c = 0.5 * (pQ2max + pQ2min);

        /*
         * Integrate the doubly differential cross-section over Q2 using
         * a Gaussian quadrature. Note that 9 points are enough to get a
         * better than 0.1 % accuracy.
        */
        const double ds =
            utils::numerics::quadrature9<double>(0.f, 1.f, [&A, &pQ2c, &dpQ2, &mass, &K, &q](const double &t) {
                const double Q2 = exp(pQ2c + 0.5 * dpQ2 * t);
                return dcs_photonuclear_d2(A, mass, K, q, Q2) * Q2;
            });

        return (ds < 0.) ? 0. : 0.5 * ds * dpQ2 * 1E+03 * AVOGADRO_NUMBER * (mass + K) / A;
    };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9620
     */
    inline const auto default_ionisation = [](const KineticEnergy &K,
                                              const RecoilEnergy &q,
                                              const AtomicElement &element,
                                              const ParticleMass &mass) {
        const double A = element.A;
        const int Z = element.Z;

        const double P2 = K * (K + 2. * mass);
        const double E = K + mass;
        const double Wmax = 2. * ELECTRON_MASS * P2 /
                            (mass * mass +
                             ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
        if ((Wmax < X_FRACTION * K) || (q > Wmax))
            return 0.;
        const double Wmin = 0.62 * element.I;
        if (q <= Wmin)
            return 0.;

        /* Close interactions for Q >> atomic binding energies. */
        const double a0 = 0.5 / P2;
        const double a1 = -1. / Wmax;
        const double a2 = E * E / P2;
        const double cs =
            1.535336E-05 * E * Z / A * (a0 + 1. / q * (a1 + a2 / q));

        /* Radiative correction. */
        double Delta = 0.;
        const double m1 = mass - ELECTRON_MASS;
        if (K >= 0.5 * m1 * m1 / ELECTRON_MASS)
        {
            const double L1 = log(1. + 2. * q / ELECTRON_MASS);
            Delta = 1.16141E-03 * L1 *
                    (log(4. * E * (E - q) / (mass * mass)) -
                     L1);
        }
        return cs * (1. + Delta);
    };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9669
     */
    inline const auto analytic_integral_ionisation = [](const KineticEnergy &K,
                                                        const EnergyTransferMin &xlow,
                                                        const AtomicElement &element,
                                                        const ParticleMass &mass,
                                                        const ComputeCEL cel = false) {
        const double P2 = K * (K + 2. * mass);
        const double E = K + mass;
        const double Wmax = 2. * ELECTRON_MASS * P2 /
                            (mass * mass +
                             ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
        if (Wmax < X_FRACTION * K)
            return 0.;
        double Wmin = 0.62 * element.I;
        const double qlow = K * xlow;
        if (qlow >= Wmin)
            Wmin = qlow;

        /* Check the bounds. */
        if (Wmax <= Wmin)
            return 0.;

        /* Close interactions for Q >> atomic binding energies. */
        const double a0 = 0.5 / P2;
        const double a1 = -1. / Wmax;
        const double a2 = E * E / P2;

        double S;
        if (!cel)
        {
            S = a0 * (Wmax - Wmin) + a1 * log(Wmax / Wmin) +
                a2 * (1. / Wmin - 1. / Wmax);
        }
        else
        {
            S = 0.5 * a0 * (Wmax * Wmax - Wmin * Wmin) +
                a1 * (Wmax - Wmin) + a2 * log(Wmax / Wmin);
        }

        return 1.535336E-05 * element.Z / element.A * S;
    };

    template <>
    inline auto compute_integral(const decltype(default_ionisation) &dcs_kernel)
    {
        return [&dcs_kernel](const KineticEnergy &K,
                             const EnergyTransferMin &xlow,
                             const AtomicElement &element,
                             const ParticleMass &mass,
                             const int min_points,
                             const ComputeCEL cel = false) {
            const double m1 = mass - ELECTRON_MASS;
            if (K <= 0.5 * m1 * m1 / ELECTRON_MASS)
                return analytic_integral_ionisation(K, xlow, element, mass, cel);
            return utils::numerics::quadrature6<double>(
                       log(K * xlow), log(K),
                       [&](const double &t) {
                           const double q = exp(t);
                           double s = dcs_kernel(K, q, element, mass) * q;
                           if (cel)
                               s *= q;
                           return s;
                       },
                       min_points) /
                   (K + mass);
        };
    }

    template <typename DELKernels>
    inline auto compute_dcs_integrals(const DELKernels &del_kernels,
                                      const Result &result,
                                      const KineticEnergies &K,
                                      const EnergyTransferMin &xlow,
                                      const AtomicElement &element,
                                      const ParticleMass &mass,
                                      const int min_points,
                                      const ComputeCEL cel = false)
    {
        const auto &[br, pp, ph, io] = del_kernels;
        map_compute_integral(br)(result[0], K, xlow, element, mass, min_points, cel);
        map_compute_integral(pp)(result[1], K, xlow, element, mass, min_points, cel);
        map_compute_integral(ph)(result[2], K, xlow, element, mass, min_points, cel);
        map_compute_integral(io)(result[3], K, xlow, element, mass, min_points, cel);
    }

    inline Result compute_be_cel(
        const Table &br,
        const Table &pp,
        const Table &ph,
        const Table &io,
        const Table &cs)
    {
        auto be_cel = torch::zeros_like(br);
        be_cel += torch::where(cs[0] < br, cs[0], br);
        be_cel += torch::where(cs[1] < pp, cs[1], pp);
        be_cel += torch::where(cs[2] < ph, cs[2], ph);
        be_cel += torch::where(cs[3] < io, cs[3], io);
        return be_cel;
    }

    inline ThresholdIndex compute_kinetic_threshold(
        const Table &table_CS,
        const Table &table_dE,
        const Table &table_NI_in)
    {
        int ri = 0;
        const int nkin = table_CS.numel();
        Scalar cs0 = 0.0;
        Scalar *cs = table_CS.data_ptr<Scalar>();
        for (ri = 0; ri < nkin; ri++)
            if ((cs0 = cs[ri]) > 0)
                break;
        Scalar *dE = table_dE.data_ptr<Scalar>();
        Scalar *NI_in = table_NI_in.data_ptr<Scalar>();
        for (int i = 0; i < ri; i++)
        {
            cs[i] = cs0;
            NI_in[i] = cs0 / dE[i];
        }
        return ri;
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8886
     */
    template <typename DELKernel>
    inline void compute_threshold(
        const DELKernel &del_kernel,
        const Thresholds &Xt,
        const KineticEnergies &K,
        const EnergyTransferMin &xlow,
        const AtomicElement &element,
        const ParticleMass &mass,
        const ThresholdIndex th_i)
    {
        int n = K.numel();
        double *pXt = Xt.data_ptr<double>();
        double *pK = K.data_ptr<double>();
        for (int i = th_i; i < n; i++)
        {
            double x = xlow;
            while ((x < 1.) && (del_kernel(pK[i], pK[i] * x, element, mass) <= 0.))
                x *= 2;
            if (x >= 1.)
                x = 1.;
            else if (x > xlow)
            {
                const double eps = 1E-02 * xlow;
                double x0 = 0.5 * x;
                double dcs = 0.;
                for (;;)
                {
                    if (dcs == 0.)
                        x0 += 0.5 * (x - x0);
                    else
                    {
                        const double dx =
                            x - x0;
                        x = x0;
                        x0 -= 0.5 * dx;
                    }
                    if ((x - x0) <= eps)
                        break;
                    dcs = del_kernel(pK[i], pK[i] * x0, element, mass);
                }
            }
            pXt[i] = x;
        }
    }

    template <typename DELKernels>
    inline void compute_fractional_thresholds(
        const DELKernels &del_kernels,
        const Thresholds &Xt,
        const KineticEnergies &K,
        const EnergyTransferMin &xlow,
        const AtomicElement &element,
        const ParticleMass &mass,
        ThresholdIndex th_i)
    {
        const auto &[br, pp, ph, io] = del_kernels;
        compute_threshold(br, Xt[0], K, xlow, element, mass, th_i);
        compute_threshold(pp, Xt[1], K, xlow, element, mass, th_i);
        compute_threshold(ph, Xt[2], K, xlow, element, mass, th_i);
        compute_threshold(io, Xt[3], K, xlow, element, mass, th_i);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6054
     */
    inline double coulomb_frame_parameters(double *fCM,
                                           const KineticEnergy &K,
                                           const AtomicElement &element,
                                           const ParticleMass &mass)
    {
        double kinetic0;
        const double Ma = element.A * ATOMIC_MASS_ENERGY;
        double M2 = mass + Ma;
        M2 *= M2;
        const double sCM12i = 1. / sqrt(M2 + 2. * Ma * K);
        fCM[0] = (K + mass + Ma) * sCM12i;
        kinetic0 =
            (K * Ma + mass * (mass + Ma)) * sCM12i -
            mass;
        if (kinetic0 < KIN_CUTOFF)
            kinetic0 = KIN_CUTOFF;
        const double etot = K + mass + Ma;
        const double betaCM2 =
            K * (K + 2. * mass) / (etot * etot);
        double rM2 = mass / Ma;
        rM2 *= rM2;
        fCM[1] = sqrt(rM2 * (1. - betaCM2) + betaCM2);
        return kinetic0;
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6038
     */
    inline double coulomb_spin_factor(const KineticEnergy &K, const ParticleMass &mass)
    {
        const double e = K + mass;
        return K * (e + mass) / (e * e);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L5992
     */
    inline double coulomb_wentzel_path(const double &screening,
                                       const KineticEnergy &K,
                                       const AtomicElement &element,
                                       const ParticleMass &mass)
    {
        const double d = K * (K + 2. * mass) /
                         (element.Z * (K + mass));
        return element.A * 2.54910918E+08 * screening * (1. + screening) * d * d;
    }
    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L5934
     */
    inline double coulomb_screening_parameters(double *pscreen,
                                               const KineticEnergy &K,
                                               const AtomicElement &element,
                                               const ParticleMass &mass)
    {
        // Nuclear screening
        const double third = 1. / 3;
        const double A13 = pow(element.A, third);
        const double R1 = 1.02934 * A13 + 0.435;
        const double R2 = 2.;
        const double p2 = K * (K + 2. * mass);
        const double d = 5.8406E-02 / p2;
        pscreen[1] = d / (R1 * R1);
        pscreen[2] = d / (R2 * R2);

        // Atomic Moliere screening with Coulomb correction from Kuraev et al.
        // Phys. Rev. D 89, 116016 (2014). Valid for ultra-relativistic
        // particles only.

        const int Z = element.Z;
        const double etot = K + mass;
        const double ZE = Z * etot;
        const double zeta2 = 5.3251346E-05 * (ZE * ZE) / p2;
        double cK;
        if (zeta2 > 1.)
        {
            // Let's perform the serie computation.
            int i, n = 10 + Z;
            double f = 0.;
            for (i = 1; i <= n; i++)
                f += zeta2 / (i * (i * i + zeta2));
            cK = exp(f);
        }
        else
        {
            // Let's use Kuraev's approximate expression.
            cK = exp(1. - 1. / (1. + zeta2) +
                     zeta2 * (0.2021 + zeta2 * (0.0083 * zeta2 - 0.0369)));
        }

        // Original Moliere's atomic screening, considered as a reference
        // value at low energies.

        const double cM = 1. + 3.34 * zeta2;

        // Atomic screening interpolation.
        double r = K / etot;
        r *= r;
        const double c = r * cK + (1. - r) * cM;
        pscreen[0] = 5.179587126E-12 * pow(Z, 2. / 3.) * c / p2;

        const double d01 = 1. / (pscreen[0] - pscreen[1]);
        const double d02 = 1. / (pscreen[0] - pscreen[2]);
        const double d12 = 1. / (pscreen[1] - pscreen[2]);
        pscreen[6] = d01 * d01 * d02 * d02;
        pscreen[7] = d01 * d01 * d12 * d12;
        pscreen[8] = d12 * d12 * d02 * d02;
        pscreen[3] = 2. * pscreen[6] * (d01 + d02);
        pscreen[4] = 2. * pscreen[7] * (d12 - d01);
        pscreen[5] = -2. * pscreen[8] * (d12 + d02);

        return 1. / coulomb_wentzel_path(pscreen[0], K, element, mass);
    }

    inline const auto default_coulomb_data =
        [](
            const CMLorentz &fCM,
            const ScreeningFactors &screening,
            const FSpins &fspin,
            const InvLambdas &invlambda,
            const KineticEnergies &K,
            const AtomicElement &element,
            const ParticleMass &mass) {
            const int nkin = K.numel();
            double *pK = K.data_ptr<double>();

            double *pfCM = fCM.data_ptr<double>();
            double *pscreen = screening.data_ptr<double>();
            double *pfspin = fspin.data_ptr<double>();
            double *pinvlbd = invlambda.data_ptr<double>();

            for (int i = 0; i < nkin; i++)
            {
                const double kinetic0 = coulomb_frame_parameters(pfCM + 2 * i, pK[i], element, mass);
                pfspin[i] = coulomb_spin_factor(kinetic0, mass);
                pinvlbd[i] = coulomb_screening_parameters(pscreen + NSF * i, kinetic0, element, mass);
            }
        };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6160
     */
    void coulomb_transport_coefficients(
        double *pcoefs,
        double *pscreen,
        const double &fspin,
        const double &mu)
    {
        const double nuclear_screening =
            (pscreen[1] < pscreen[2]) ? pscreen[1] : pscreen[2];
        if (mu < 1E-08 * nuclear_screening)
        {
            // We neglect the nucleus finite size.
            const double L = log(1. + mu / pscreen[0]);
            const double r = mu / (mu + pscreen[0]);
            const double k = pscreen[0] * (1. + pscreen[0]);
            pcoefs[0] = k * (r / pscreen[0] - fspin * (L - r));
            const double I2 = mu - pscreen[0] * (r - 2. * L);
            pcoefs[1] = 2. * k * (L - r - fspin * I2);
        }
        else
        {
            // We need to take all factors into account using a pole reduction.
            double I0[3], I1[3], I2[3], J0[3], J1[3], J2[3];
            int i;
            double mu2 = 0.5 * mu * mu;
            for (i = 0; i < 3; i++)
            {
                double r = mu / (mu + pscreen[i]);
                double L = log(1. + mu / pscreen[i]);
                double mu1 = mu;
                I0[i] = r / pscreen[i];
                J0[i] = L;
                I1[i] = L - r;
                r *= pscreen[i];
                L *= pscreen[i];
                J1[i] = mu1 - L;
                I2[i] = mu1 - 2. * L + r;
                L *= pscreen[i];
                mu1 *= pscreen[i];
                J2[i] = mu2 + L - mu1;
            }

            const double k = pscreen[0] * (1. + pscreen[0]) *
                             pscreen[1] * pscreen[1] * pscreen[2] * pscreen[2];
            pcoefs[0] = pcoefs[1] = 0.;
            for (i = 0; i < 3; i++)
            {
                pcoefs[0] += pscreen[3 + i] * (J0[i] - fspin * J1[i]) +
                             pscreen[6 + i] * (I0[i] - fspin * I1[i]);
                pcoefs[1] += pscreen[3 + i] * (J1[i] - fspin * J2[i]) +
                             pscreen[6 + i] * (I1[i] - fspin * I2[i]);
            }
            pcoefs[0] *= k;
            pcoefs[1] *= 2. * k;
        }
    }

    inline const auto default_coulomb_transport =
        [](
            const TransportCoefs &coefficients,
            const ScreeningFactors &screening,
            const FSpins &fspin,
            const AngularCutoff &mu) {
            double *pcoefs = coefficients.data_ptr<double>();
            double *pscreen = screening.data_ptr<double>();
            double *pfspin = fspin.data_ptr<double>();

            const bool nmu = (mu.numel() == 1);
            double *pmu = mu.data_ptr<double>();

            const int nspin = fspin.numel();
            for (int i = 0; i < nspin; i++)
                coulomb_transport_coefficients(
                    pcoefs + 2 * i,
                    pscreen + NSF * i,
                    pfspin[i],
                    pmu[(nmu) ? 0 : i]);
        };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6109
     */
    inline double coulomb_restricted_cs(
        const double &mu,
        const double &fspin,
        double *screen)
    {
        if (mu >= 1.)
            return 0.;

        const double nuclear_screening =
            (screen[1] < screen[2]) ? screen[1] : screen[2];
        if (mu < 1E-08 * nuclear_screening)
        {
            // We neglect the nucleus finite size.
            const double L = log((screen[0] + 1.) / (screen[0] + mu));
            const double r =
                (1. - mu) / ((screen[0] + mu) * (screen[0] + 1.));
            const double k = screen[0] * (1. + screen[0]);
            return k * (r - fspin * (L - screen[0] * r));
        }
        else
        {
            // We need to take all factors into account using a pole reduction.
            double I0[3], I1[3], J0[3], J1[3];
            int i;
            for (i = 0; i < 3; i++)
            {
                const double L =
                    log((screen[i] + 1.) / (screen[i] + mu));
                const double r = (1. - mu) /
                                 ((screen[i] + mu) * (screen[i] + 1.));
                I0[i] = r;
                J0[i] = L;
                I1[i] = L - screen[i] * r;
                J1[i] = mu - screen[i] * L;
            }

            const double k = screen[0] * (1. + screen[0]) *
                             screen[1] * screen[1] * screen[2] * screen[2];
            double cs = 0.;
            for (i = 0; i < 3; i++)
            {
                cs += screen[3 + i] * (J0[i] - fspin * J1[i]) +
                      screen[6 + i] * (I0[i] - fspin * I1[i]);
            }
            return k * cs;
        }
    }

    inline double cutoff_objective(
        const double &cs_h,
        const double &mu,
        double *invlambda,
        double *fspin,
        double *screen,
        const int nel = 1,
        const int nkin = 1)
    {
        double cs_tot = 0.;
        for (int iel = 0; iel < nel; iel++)
        {
            const int off = iel * nkin;
            cs_tot += invlambda[off] * coulomb_restricted_cs(mu, fspin[off], screen + NSF * off);
        }
        return cs_tot - cs_h;
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8472
     */
    inline void coulomb_hard_scattering(double &mu0, double &lb_h,
                                        double *G, double *fCM,
                                        double *screen,
                                        double *invlambda,
                                        double *fspin,
                                        const int nel = 1,
                                        const int nkin = 1)
    {

        double invlb_m = 0., invlb1_m = 0.;
        double s_m_l = 0., s_m_h = 0.;

        for (int iel = 0; iel < nel; iel++)
        {
            const int off = iel * nkin;

            const double invlb = invlambda[off];
            const double scr = screen[NSF * off];

            invlb_m += invlb * G[2 * off];
            s_m_h += scr * invlb;
            s_m_l += invlb / scr;
            const double d = 1. / (fCM[2 * off] * (1. + fCM[1 + 2 * off]));
            invlb1_m += invlb * G[1 + 2 * off] * d * d;
        }

        // Set the hard scattering mean free path.
        const double lb_m = 1. / invlb_m;
        lb_h = std::min(EHS_OVER_MSC / invlb1_m, EHS_PATH_MAX);

        // Compute the hard scattering cutoff angle, in the CM.
        if (lb_m < lb_h)
        {
            // Initialise the root finder with an asymptotic starting value
            // Asymptotic value when lb_h >> lb_m versus when lb_h ~= lb_m
            double s_m = (lb_h > 2. * lb_m) ? s_m_h * lb_m : 1. / (s_m_l * lb_m);

            mu0 = s_m * (lb_h - lb_m) / (s_m * lb_h + lb_m);

            // targeted cross section
            const double cs_h = 1. / lb_h;

            // Configure for the root solver.
            // Solve for the cut-off angle. We try an initial bracketing in
            // [0.25*mu0; 4.*mu0], with mu0 the asymptotic estimate.
            double mu_max = std::min(4. * mu0, 1.);
            double mu_min = 0.25 * mu0;

            double fmax = 0, fmin = 0;

            fmax = cutoff_objective(cs_h, mu_max, invlambda, fspin, screen, nel, nkin);
            if (fmax > 0.)
            {
                // This shouldn't occur, but let's be safe and handle this case.
                mu_min = mu_max;
                fmin = fmax;
                mu_max = 1.;
                fmax = -cs_h;
            }
            else
            {
                fmin = cutoff_objective(cs_h, mu_min, invlambda, fspin, screen, nel, nkin);
                if (fmin < 0.)
                {
                    // This might occur at high energies when the nuclear screening becomes significant.
                    mu_max = mu_min;
                    fmax = fmin;
                    mu_min = 0.;
                    fmin = cutoff_objective(cs_h, mu_min, invlambda, fspin, screen, nel, nkin);
                }
                if (mu_min < MAX_MU0)
                {
                    mu_max = std::min(mu_max, MAX_MU0);
                    const auto mubest =
                        utils::numerics::ridders_root<double>(
                            mu_min, mu_max,
                            [&](const double &mu_x) {
                                return cutoff_objective(cs_h, mu_x, invlambda, fspin, screen, nel, nkin);
                            },
                            fmin, fmax,
                            1E-6 * mu0, 1E-6, 100);

                    if (mubest.has_value())
                        mu0 = mubest.value();
                }
                mu0 = std::min(mu0, MAX_MU0);
                lb_h = cutoff_objective(cs_h, mu0, invlambda, fspin, screen, nel, nkin) + cs_h;
                lb_h = (lb_h <= 1. / EHS_PATH_MAX) ? EHS_PATH_MAX : 1. / lb_h;
            }
        }
        else
        {
            lb_h = lb_m;
            mu0 = 0;
        }
    }

    inline const auto default_hard_scattering =
        [](const AngularCutoff &mu0,
           const HSMeanFreePath &lb_h,
           const TransportCoefs &coefficients,
           const CMLorentz &transform,
           const ScreeningFactors &screening,
           const InvLambdas &invlambdas,
           const FSpins &fspins) {
            const int nel = invlambdas.size(0);
            const int nkin = invlambdas.size(1);

            double *pmu0 = mu0.data_ptr<double>();
            double *plb_h = lb_h.data_ptr<double>();

            double *invlambda = invlambdas.data_ptr<double>();
            double *fspin = fspins.data_ptr<double>();
            double *G = coefficients.data_ptr<double>();
            double *fCM = transform.data_ptr<double>();
            double *screen = screening.data_ptr<double>();

            for (int i = 0; i < nkin; i++)
                coulomb_hard_scattering(
                    pmu0[i],
                    plb_h[i],
                    G + 2 * i,
                    fCM + 2 * i,
                    screen + NSF * i,
                    invlambda + i,
                    fspin + i,
                    nel, nkin);
        };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6223
     */
    inline double transverse_transport_ionisation(
        const KineticEnergy &K,
        const AtomicElement &element,
        const ParticleMass &mass)
    {
        // Soft close interactions, restricted to X_FRACTION.
        const double momentum2 = K * (K + 2. * mass);
        const double E = K + mass;
        const double Wmax = 2. * ELECTRON_MASS * momentum2 /
                            (mass * mass +
                             ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
        const double W0 = 2. * momentum2 / ELECTRON_MASS;
        const double mu_max = Wmax / W0;
        double mu3 = K * X_FRACTION / W0;
        if (mu3 > mu_max)
            mu3 = mu_max;
        const double mu2 = 0.62 * element.I / W0;
        if (mu2 >= mu3)
            return 0.;
        const double a0 = 0.5 * W0 / momentum2;
        const double a1 = -1. / Wmax;
        const double a2 = E * E / (W0 * momentum2);
        const double cs0 = 1.535336E-05 / element.A; /* m^2/kg/GeV. */
        return 2. * cs0 * element.Z *
               (0.5 * a0 * (mu3 * mu3 - mu2 * mu2) + a1 * (mu3 - mu2) +
                a2 * log(mu3 / mu2));
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6262
     */
    inline double transverse_transport_photonuclear(
        const KineticEnergy &K,
        const AtomicElement &element,
        const ParticleMass &mass)
    {
        // Integration over the K transfer, q, done with a log sampling.
        const double E = K + mass;
        return 2. * utils::numerics::quadrature6<double>(
                        log(1E-06), 0.,
                        [&](const double &t) {
                            const double nu = X_FRACTION * exp(t);
                            const double q = nu * K;

                            // Analytical integration over mu.
                            const double m02 = 0.4;
                            const double q2 = q * q;
                            const double tmax = 1.876544 * q;
                            const double tmin =
                                q2 * mass * mass / (E * (E - q));
                            const double b1 = 1. / (1. - q2 / m02);
                            const double c1 = 1. / (1. - m02 / q2);
                            double L1 = b1 * log((q2 + tmax) / (q2 + tmin));
                            double L2 = c1 * log((m02 + tmax) / (m02 + tmin));
                            const double I0 = log(tmax / tmin) - L1 - L2;
                            L1 *= q2;
                            L2 *= m02;
                            const double I1 = L1 + L2;
                            L1 *= q2;
                            L2 *= m02;
                            const double I2 =
                                (tmax - tmin) * (b1 * q2 + c1 * m02) - L1 - L2;
                            const double ratio =
                                (I1 * tmax - I2) / ((I0 * tmax - I1) * K *
                                                    (K + 2. * mass));

                            return default_photonuclear(K, q, element, mass) * ratio * nu;
                        },
                        100);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8730
     */
    inline const auto default_soft_scattering =
        [](const Result &ms1,
           const KineticEnergies &K,
           const AtomicElement &element,
           const ParticleMass &mass) {
            utils::vmap<double>(
                K,
                [&](const double &k) { return transverse_transport_ionisation(k, element, mass) +
                                              transverse_transport_photonuclear(k, element, mass); },
                ms1);
        };

    inline const auto default_del_kernels = std::tuple{
        default_bremsstrahlung,
        default_pair_production,
        default_photonuclear,
        default_ionisation};

    inline const auto default_tt_kernels = std::tuple{
        default_coulomb_data,
        default_coulomb_transport,
        default_hard_scattering,
        default_soft_scattering};

    inline const auto default_kernels = std::tuple{
        default_del_kernels,
        default_tt_kernels};

} // namespace ghmc::pms::dcs
