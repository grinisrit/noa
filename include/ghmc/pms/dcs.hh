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
    using KineticEnergy = Scalar;
    using RecoilEnergy = Scalar;
    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using Result = torch::Tensor; // Receiver tensor for calculations result
    using ComputeCEL = bool;      // Compute Continuous Energy Loss (CEL) flag
    using CSTab = torch::Tensor;
    using CSComputed = torch::Tensor;
    using ThresholdIndex = Index;
    using Thresholds = torch::Tensor;

    template <typename DCSKernel>
    inline auto map_kernel(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const Result &result,
                             const KineticEnergies &K,
                             const RecoilEnergies &q,
                             const AtomicElement &element,
                             const ParticleMass &mu) {
            const double *pq = q.data_ptr<double>();
            int i = 0;
            ghmc::utils::vmap<double>(result, K, [&](const auto &k) {
                return dcs_kernel(k, pq[i++], element, mu);
            });
        };
    }

    template <typename DCSKernel>
    inline auto compute_integral(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const KineticEnergy &K,
                             const EnergyTransferMin &xlow,
                             const AtomicElement &element,
                             const ParticleMass &mu,
                             const int min_points,
                             const ComputeCEL cel = false) {
            return ghmc::numerics::quadrature6<double>(
                       log(K * xlow), log(K),
                       [&](const double &t) {
                           const double q = exp(t);
                           double s = dcs_kernel(K, q, element, mu) * q;
                           if (cel)
                               s *= q;
                           return s;
                       },
                       min_points) /
                   (K + mu);
        };
    }

    template <typename DCSKernel>
    inline auto map_compute_integral(const DCSKernel &dcs_kernel)
    {
        return [&dcs_kernel](const Result &result,
                             const KineticEnergies &K,
                             const EnergyTransferMin &xlow,
                             const AtomicElement &element,
                             const ParticleMass &mu,
                             const int min_points,
                             const ComputeCEL cel = false) {
            ghmc::utils::vmap<double>(result, K, [&](const double &k) {
                return compute_integral(dcs_kernel)(
                    k, xlow, element, mu, min_points, cel);
            });
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
                                                  const ParticleMass &mu) {
        const int Z = element.Z;
        const auto A = element.A;
        const double me = ELECTRON_MASS;
        const double sqrte = 1.648721271;
        const double phie_factor = mu / (me * me * sqrte);
        const double rem = 5.63588E-13 * me / mu;

        const double BZ_n = (Z == 1) ? 202.4 : 182.7 * pow(Z, -1. / 3.);
        const double BZ_e = (Z == 1) ? 446. : 1429. * pow(Z, -2. / 3.);
        const double D_n = 1.54 * pow(A, 0.27);
        const double E = K + mu;
        const double dcs_factor = 7.297182E-07 * rem * rem * Z / E;

        const double delta_factor = 0.5 * mu * mu / E;
        const double qe_max = E / (1. + 0.5 * mu * mu / (me * E));

        const double nu = q / E;
        const double delta = delta_factor * nu / (1. - nu);
        double Phi_n, Phi_e;
        Phi_n = log(BZ_n * (mu + delta * (D_n * sqrte - 2.)) /
                    (D_n * (me + delta * sqrte * BZ_n)));
        if (Phi_n < 0.)
            Phi_n = 0.;
        if (q < qe_max)
        {
            Phi_e = log(BZ_e * mu /
                        ((1. + delta * phie_factor) * (me + delta * sqrte * BZ_e)));
            if (Phi_e < 0.)
                Phi_e = 0.;
        }
        else
            Phi_e = 0.;

        const double dcs =
            dcs_factor * (Z * Phi_n + Phi_e) * (4. / 3. * (1. / nu - 1.) + nu);
        return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mu + K) / A;
    };

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
     */
    inline const auto default_pair_production = [](const KineticEnergy &K,
                                                   const RecoilEnergy &q,
                                                   const AtomicElement &element,
                                                   const ParticleMass &mu) {
        const int Z = element.Z;
        const double A = element.A;
        /*  Check the bounds of the energy transfer. */
        if (q <= 4. * ELECTRON_MASS)
            return 0.;
        const double sqrte = 1.6487212707;
        const double Z13 = pow(Z, 1. / 3.);
        if (q >= K + mu * (1. - 0.75 * sqrte * Z13))
            return 0.;

        /*  Precompute some constant factors for the compute_integral. */
        const double nu = q / (K + mu);
        const double r = mu / ELECTRON_MASS;
        const double beta = 0.5 * nu * nu / (1. - nu);
        const double xi_factor = 0.5 * r * r * beta;
        const double A_ = (Z == 1) ? 202.4 : 183.;
        const double AZ13 = A_ / Z13;
        const double cL = 2. * sqrte * ELECTRON_MASS * AZ13;
        const double cLe = 2.25 * Z13 * Z13 / (r * r);

        /*  Compute the bound for the integral. */
        const double gamma = 1. + K / mu;
        const double x0 = 4. * ELECTRON_MASS / q;
        const double x1 = 6. / (gamma * (gamma - q / mu));
        const double argmin =
            (x0 + 2. * (1. - x0) * x1) / (1. + (1. - x1) * sqrt(1. - x0));
        if ((argmin >= 1.) || (argmin <= 0.))
            return 0.;
        const double tmin = log(argmin);

        /*  Compute the integral over t = ln(1-rho). */
        double I = ghmc::numerics::quadrature8<double>(0.f, 1.f, [&](const double &t) {
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

            /* Compute the mu-term. */
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
        const double E = K + mu;
        const double dcs = 1.794664E-34 * Z * (Z + zeta) * (E - q) * I /
                           (q * E);
        return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mu + K) / A;
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

    inline double dcs_photonuclear_d2(const double A, const double mu, const double K, const double q, const double Q2)
    {
        const double cf = 2.603096E-35;
        const double M = 0.931494;
        const double E = K + mu;

        const double y = q / E;
        const double x = 0.5 * Q2 / (M * q);
        const double F2p = dcs_photonuclear_f2_allm(x, Q2);
        const double F2A = dcs_photonuclear_f2a_drss(x, F2p, A);
        const double R = dcs_photonuclear_r_whitlow(x, Q2);

        const double dds = (1 - y +
                            0.5 * (1 - 2 * mu * mu / Q2) *
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
                                                const ParticleMass &mu) {
        if (dcs_photonuclear_check(K, q))
            return 0.;

        const double A = element.A;
        const double mass = mu;
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
            ghmc::numerics::quadrature9<double>(0.f, 1.f, [&A, &pQ2c, &dpQ2, &mass, &K, &q](const double &t) {
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
                                              const ParticleMass &mu) {
        const double A = element.A;
        const int Z = element.Z;
        const double mass = mu;

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
                                                        const ParticleMass &mu,
                                                        const ComputeCEL cel = false) {
        const double mass = mu;
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
                             const ParticleMass &mu,
                             const int min_points,
                             const ComputeCEL cel = false) {
            const double m1 = mu - ELECTRON_MASS;
            if (K <= 0.5 * m1 * m1 / ELECTRON_MASS)
                return analytic_integral_ionisation(K, xlow, element, mu, cel);
            return ghmc::numerics::quadrature6<double>(
                       log(K * xlow), log(K),
                       [&](const double &t) {
                           const double q = exp(t);
                           double s = dcs_kernel(K, q, element, mu) * q;
                           if (cel)
                               s *= q;
                           return s;
                       },
                       min_points) /
                   (K + mu);
        };
    }

    constexpr int NPR = 4; // Number of DEL processes considered

    inline const auto default_kernels = std::tuple{
        default_bremsstrahlung,
        default_pair_production,
        default_photonuclear,
        default_ionisation};

    template <typename DCSKernels>
    inline auto compute_dcs_integrals(const DCSKernels &dcs_kernels,
                                      const Result &result,
                                      const KineticEnergies &K,
                                      const EnergyTransferMin &xlow,
                                      const AtomicElement &element,
                                      const ParticleMass &mu,
                                      const int min_points,
                                      const ComputeCEL cel = false)
    {
        const auto &[br, pp, ph, io] = dcs_kernels;
        map_compute_integral(br)(result[0], K, xlow, element, mu, min_points, cel);
        map_compute_integral(pp)(result[1], K, xlow, element, mu, min_points, cel);
        map_compute_integral(ph)(result[2], K, xlow, element, mu, min_points, cel);
        map_compute_integral(io)(result[3], K, xlow, element, mu, min_points, cel);
    }

    inline Result compute_be_cel(
        const CSTab &br,
        const CSTab &pp,
        const CSTab &ph,
        const CSTab &io,
        const CSComputed &cs)
    {
        auto be_cel = torch::zeros_like(br);
        be_cel += torch::where(cs[0] < br, cs[0], br);
        be_cel += torch::where(cs[1] < pp, cs[1], pp);
        be_cel += torch::where(cs[2] < ph, cs[2], ph);
        be_cel += torch::where(cs[3] < io, cs[3], io);
        return be_cel;
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8886
     */
    template <typename DSCKernel>
    inline void compute_threshold(
        const DSCKernel &dcs_kernel,
        const Thresholds &Xt,
        const KineticEnergies &K,
        const EnergyTransferMin &xlow,
        const AtomicElement &element,
        const ParticleMass &mu,
        ThresholdIndex th_i)
    {
        int n = K.numel();
        double *pXt = Xt.data_ptr<double>();
        double *pK = K.data_ptr<double>();
        for (int i = th_i; i < n; i++)
        {
            double x = xlow;
            while ((x < 1.) && (dcs_kernel(pK[i], pK[i] * x, element, mu) <= 0.))
                x *= 2;
            if (x >= 1.)
                x = 1.;
            else if (x > X_FRACTION)
            {
                const double eps = 1E-02 * X_FRACTION;
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
                    dcs = dcs_kernel(pK[i], pK[i] * x0, element, mu);
                }
            }
            pXt[i] = x;
        }
    }

    template <typename DCSKernels>
    inline void compute_fractional_thresholds(
        const DCSKernels &dcs_kernels,
        const Thresholds &Xt,
        const KineticEnergies &K,
        const EnergyTransferMin &xlow,
        const AtomicElement &element,
        const ParticleMass &mu,
        ThresholdIndex th_i)
    {
        const auto &[br, pp, ph, io] = dcs_kernels;
        compute_threshold(br, Xt[0], K, xlow, element, mu, th_i);
        compute_threshold(pp, Xt[1], K, xlow, element, mu, th_i);
        compute_threshold(ph, Xt[2], K, xlow, element, mu, th_i);
        compute_threshold(io, Xt[3], K, xlow, element, mu, th_i);
    }

    /*
     *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
     *  GNU Lesser General Public License version 3
     *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6054
     */
    KineticEnergy coulomb_frame_parameters(const CMLorentz &fCM,
                                           const KineticEnergy &kinetic,
                                           const AtomicMass &Ma,
                                           const ParticleMass &mu,
                                           const KineticEnergy &cutoff)
    {
        double kinetic0;
        double *parameters = fCM.data_ptr<double>();
        double M2 = mu + Ma;
        M2 *= M2;
        const double sCM12i = 1. / sqrt(M2 + 2. * Ma * kinetic);
        parameters[0] = (kinetic + mu + Ma) * sCM12i;
        kinetic0 =
            (kinetic * Ma + mu * (mu + Ma)) * sCM12i -
            mu;
        if (kinetic0 < cutoff)
            kinetic0 = cutoff;
        const double etot = kinetic + mu + Ma;
        const double betaCM2 =
            kinetic * (kinetic + 2. * mu) / (etot * etot);
        double rM2 = mu / Ma;
        rM2 *= rM2;
        parameters[1] = sqrt(rM2 * (1. - betaCM2) + betaCM2);
        return kinetic0;
    }

} // namespace ghmc::pms::dcs
