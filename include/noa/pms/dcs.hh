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
#include "noa/utils/numerics.hh"

#include <torch/types.h>

namespace noa::pms::dcs {

    template<typename Dtype, typename DCSFunc>
    inline auto vmap(const DCSFunc &dcs_func) {
        return [&dcs_func](const Calculation &result,
                           const Energies &kinetic_energies,
                           const Energies &recoil_energies,
                           const AtomicElement<Dtype> &element,
                           const Dtype &mass) {
            const Dtype *recoil_energy = recoil_energies.data_ptr<Dtype>();
            utils::vmapi<Dtype>(
                    kinetic_energies,
                    [&](const int64_t i, const auto &k) { return dcs_func(k, recoil_energy[i], element, mass); },
                    result);
        };
    }

    template<typename Dtype, typename DCSFunc>
    inline auto pvmap(const DCSFunc &dcs_func) {
        return [&dcs_func](const Calculation &result,
                           const Energies &kinetic_energies,
                           const Energies &recoil_energies,
                           const AtomicElement<Dtype> &element,
                           const Dtype &mass) {
            const Dtype *recoil_energy = recoil_energies.data_ptr<Dtype>();
            utils::pvmapi<Dtype>(
                    kinetic_energies,
                    [&](const int64_t i, const auto &k) { return dcs_func(k, recoil_energy[i], element, mass); },
                    result);
        };
    }

    template<typename Dtype, typename DCSIntegrand>
    inline auto recoil_energy_integral(const DCSIntegrand &dcs_integrand) {
        return [&dcs_integrand](const Dtype &kinetic_energy,
                                const Dtype &xlow,
                                const AtomicElement<Dtype> &element,
                                const Dtype &mass,
                                const Index min_points) {
            return utils::numerics::quadrature6<Dtype>(
                    log(kinetic_energy * xlow), log(kinetic_energy),
                    [&](const Dtype &t) {
                        return dcs_integrand(kinetic_energy, exp(t), element, mass);
                    },
                    min_points) /
                   (kinetic_energy + mass);
        };
    }

    template<typename Dtype, typename DCSFunc>
    inline auto cel_integral(const DCSFunc &dcs_func) {
        return recoil_energy_integral<Dtype>(
                [&dcs_func](const Dtype &k,
                            const Dtype &q,
                            const AtomicElement<Dtype> &el,
                            const Dtype &mass) {
                    return dcs_func(k, q, el, mass) * q * q;
                });
    }

    template<typename Dtype, typename DCSFunc>
    inline auto del_integral(const DCSFunc &dcs_func) {
        return recoil_energy_integral<Dtype>(
                [&dcs_func](const Dtype &k,
                            const Dtype &q,
                            const AtomicElement<Dtype> &el,
                            const Dtype &mass) {
                    return dcs_func(k, q, el, mass) * q;
                });
    }


    template<typename Dtype, typename CSIntegral>
    inline auto vmap_integral(const CSIntegral &cs_integral) {
        return [&cs_integral](const Calculation &result,
                              const Energies &kinetic_energies,
                              const EnergyTransfer &xlow,
                              const AtomicElement<Dtype> &element,
                              const ParticleMass &mass,
                              const Index min_points) {
            utils::vmap<Dtype>(
                    kinetic_energies,
                    [&](const Dtype &k) {
                        return cs_integral(k, xlow, element, mass, min_points);
                    },
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
                const Energy &kinetic_energy,
                const Energy &recoil_energy,
                const AtomicElement<Scalar> &element,
                const ParticleMass &mass) {
            const Index Z = element.Z;
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

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
         */
        inline const auto pair_production = [](const Energy &kinetic_energy,
                                               const Energy &recoil_energy,
                                               const AtomicElement<Scalar> &element,
                                               const ParticleMass &mass) {
            const Index Z = element.Z;
            const Scalar A = element.A;
            // Check the bounds of the energy transfer
            if (recoil_energy <= 4. * ELECTRON_MASS)
                return 0.;
            const Scalar sqrte = 1.6487212707;
            const Scalar Z13 = pow(Z, 1. / 3.);
            if (recoil_energy >= kinetic_energy + mass * (1. - 0.75 * sqrte * Z13))
                return 0.;

            // Precompute some constant factors for the compute_integral
            const Scalar nu = recoil_energy / (kinetic_energy + mass);
            const Scalar r = mass / ELECTRON_MASS;
            const Scalar beta = 0.5 * nu * nu / (1. - nu);
            const Scalar xi_factor = 0.5 * r * r * beta;
            const Scalar A_ = (Z == 1) ? 202.4 : 183.;
            const Scalar AZ13 = A_ / Z13;
            const Scalar cL = 2. * sqrte * ELECTRON_MASS * AZ13;
            const Scalar cLe = 2.25 * Z13 * Z13 / (r * r);

            // Compute the bound for the integral
            const Scalar gamma = 1. + kinetic_energy / mass;
            const Scalar x0 = 4. * ELECTRON_MASS / recoil_energy;
            const Scalar x1 = 6. / (gamma * (gamma - recoil_energy / mass));
            const Scalar argmin =
                    (x0 + 2. * (1. - x0) * x1) / (1. + (1. - x1) * sqrt(1. - x0));
            if ((argmin >= 1.) || (argmin <= 0.))
                return 0.;
            const Scalar tmin = log(argmin);

            // Compute the integral over t = ln(1-rho)
            const auto I = utils::numerics::quadrature8<Scalar>(0.f, 1.f, [&](const Scalar &t) {
                const Scalar eps = exp(t * tmin);
                const Scalar rho = 1. - eps;
                const Scalar rho2 = rho * rho;
                const Scalar rho21 = eps * (2. - eps);
                const Scalar xi = xi_factor * rho21;
                const Scalar xi_i = 1. / xi;

                // Compute the e-term
                Scalar Be;
                if (xi >= 1E+03)
                    Be =
                            0.5 * xi_i * ((3 - rho2) + 2. * beta * (1. + rho2));
                else
                    Be = ((2. + rho2) * (1. + beta) + xi * (3. + rho2)) *
                         log(1. + xi_i) +
                         (rho21 - beta) / (1. + xi) - 3. - rho2;
                const Scalar Ye = (5. - rho2 + 4. * beta * (1. + rho2)) /
                                  (2. * (1. + 3. * beta) * log(3. + xi_i) - rho2 -
                                   2. * beta * (2. - rho2));
                const Scalar xe = (1. + xi) * (1. + Ye);
                const Scalar cLi = cL / rho21;
                const Scalar Le = log(AZ13 * sqrt(xe) * recoil_energy / (recoil_energy + cLi * xe)) -
                                  0.5 * log(1. + cLe * xe);
                Scalar Phi_e = Be * Le;
                if (Phi_e < 0.)
                    Phi_e = 0.;

                // Compute the mass-term.
                Scalar Bmu;
                if (xi <= 1E-03)
                    Bmu = 0.5 * xi * (5. - rho2 + beta * (3. + rho2));
                else
                    Bmu = ((1. + rho2) * (1. + 1.5 * beta) -
                           xi_i * (1. + 2. * beta) * rho21) *
                          log(1. + xi) +
                          xi * (rho21 - beta) / (1. + xi) +
                          (1. + 2. * beta) * rho21;
                const Scalar Ymu = (4. + rho2 + 3. * beta * (1. + rho2)) /
                                   ((1. + rho2) * (1.5 + 2. * beta) * log(3. + xi) + 1. -
                                    1.5 * rho2);
                const Scalar xmu = (1. + xi) * (1. + Ymu);
                const Scalar Lmu =
                        log(r * AZ13 * recoil_energy / (1.5 * Z13 * (recoil_energy + cLi * xmu)));
                Scalar Phi_mu = Bmu * Lmu;
                if (Phi_mu < 0.)
                    Phi_mu = 0.;
                return -(Phi_e + Phi_mu / (r * r)) * (1. - rho) * tmin;
            });

            // Atomic electrons form factor
            Scalar zeta;
            if (gamma <= 35.)
                zeta = 0.;
            else {
                Scalar gamma1, gamma2;
                if (Z == 1.) {
                    gamma1 = 4.4E-05;
                    gamma2 = 4.8E-05;
                } else {
                    gamma1 = 1.95E-05;
                    gamma2 = 5.30E-05;
                }
                zeta = 0.073 * log(gamma / (1. + gamma1 * gamma * Z13 * Z13)) -
                       0.26;
                if (zeta <= 0.)
                    zeta = 0.;
                else {
                    zeta /=
                            0.058 * log(gamma / (1. + gamma2 * gamma * Z13)) -
                            0.14;
                }
            }

            // Gather the results and return the macroscopic DCS
            const Scalar E = kinetic_energy + mass;
            const Scalar dcs = 1.794664E-34 * Z * (Z + zeta) * (E - recoil_energy) * I /
                               (recoil_energy * E);
            return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER * (mass + kinetic_energy) / A;
        };

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9371
         */
        inline Scalar dcs_photonuclear_f2_allm(const Scalar x, const Scalar Q2) {
            const Scalar m02 = 0.31985;
            const Scalar mP2 = 49.457;
            const Scalar mR2 = 0.15052;
            const Scalar Q02 = 0.52544;
            const Scalar Lambda2 = 0.06527;

            const Scalar cP1 = 0.28067;
            const Scalar cP2 = 0.22291;
            const Scalar cP3 = 2.1979;
            const Scalar aP1 = -0.0808;
            const Scalar aP2 = -0.44812;
            const Scalar aP3 = 1.1709;
            const Scalar bP1 = 0.36292;
            const Scalar bP2 = 1.8917;
            const Scalar bP3 = 1.8439;

            const Scalar cR1 = 0.80107;
            const Scalar cR2 = 0.97307;
            const Scalar cR3 = 3.4942;
            const Scalar aR1 = 0.58400;
            const Scalar aR2 = 0.37888;
            const Scalar aR3 = 2.6063;
            const Scalar bR1 = 0.01147;
            const Scalar bR2 = 3.7582;
            const Scalar bR3 = 0.49338;

            const Scalar M2 = 0.8803505929;
            const Scalar W2 = M2 + Q2 * (1.0 / x - 1.0);
            const Scalar t = log(log((Q2 + Q02) / Lambda2) / log(Q02 / Lambda2));
            const Scalar xP = (Q2 + mP2) / (Q2 + mP2 + W2 - M2);
            const Scalar xR = (Q2 + mR2) / (Q2 + mR2 + W2 - M2);
            const Scalar lnt = log(t);
            const Scalar cP =
                    cP1 + (cP1 - cP2) * (1.0 / (1.0 + exp(cP3 * lnt)) - 1.0);
            const Scalar aP =
                    aP1 + (aP1 - aP2) * (1.0 / (1.0 + exp(aP3 * lnt)) - 1.0);
            const Scalar bP = bP1 + bP2 * exp(bP3 * lnt);
            const Scalar cR = cR1 + cR2 * exp(cR3 * lnt);
            const Scalar aR = aR1 + aR2 * exp(aR3 * lnt);
            const Scalar bR = bR1 + bR2 * exp(bR3 * lnt);

            const Scalar F2P = cP * exp(aP * log(xP) + bP * log(1 - x));
            const Scalar F2R = cR * exp(aR * log(xR) + bR * log(1 - x));

            return Q2 / (Q2 + m02) * (F2P + F2R);
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9433
         */
        inline Scalar dcs_photonuclear_f2a_drss(const Scalar x, const Scalar F2p, const Scalar A) {
            Scalar a = 1.0;
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
        inline Scalar dcs_photonuclear_r_whitlow(const Scalar x, const Scalar Q2) {
            Scalar q2 = Q2;
            if (Q2 < 0.3)
                q2 = 0.3;

            const Scalar theta =
                    1 + 12.0 * q2 / (1.0 + q2) * 0.015625 / (0.015625 + x * x);

            return (0.635 / log(q2 / 0.04) * theta + 0.5747 / q2 -
                    0.3534 / (0.09 + q2 * q2));
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9478
         */

        inline Scalar
        dcs_photonuclear_d2(const Scalar A, const Scalar mass, const Scalar kinetic_energy, const Scalar recoil_energy,
                            const Scalar Q2) {
            const Scalar cf = 2.603096E-35;
            const Scalar M = 0.931494;
            const Scalar E = kinetic_energy + mass;

            const Scalar y = recoil_energy / E;
            const Scalar x = 0.5 * Q2 / (M * recoil_energy);
            const Scalar F2p = dcs_photonuclear_f2_allm(x, Q2);
            const Scalar F2A = dcs_photonuclear_f2a_drss(x, F2p, A);
            const Scalar R = dcs_photonuclear_r_whitlow(x, Q2);

            const Scalar dds = (1 - y +
                                0.5 * (1 - 2 * mass * mass / Q2) *
                                (y * y + Q2 / (E * E)) / (1 + R)) /
                               (Q2 * Q2) -
                               0.25 / (E * E * Q2);

            return cf * F2A * dds / recoil_energy;
        }

        inline bool dcs_photonuclear_check(const Scalar kinetic_energy, const Scalar recoil_energy) {
            return (recoil_energy < 1.) || (recoil_energy < 2E-03 * kinetic_energy);
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9515
         */
        inline const auto photonuclear = [](const Energy &kinetic_energy,
                                            const Energy &recoil_energy,
                                            const AtomicElement<Scalar> &element,
                                            const ParticleMass &mass) {
            if (dcs_photonuclear_check(kinetic_energy, recoil_energy))
                return 0.;

            const Scalar A = element.A;
            const Scalar M = 0.931494;
            const Scalar mpi = 0.134977;
            const Scalar E = kinetic_energy + mass;

            if ((recoil_energy >= (E - mass)) || (recoil_energy <= (mpi * (1.0 + 0.5 * mpi / M))))
                return 0.;

            const Scalar y = recoil_energy / E;
            const Scalar Q2min = mass * mass * y * y / (1 - y);
            const Scalar Q2max = 2.0 * M * (recoil_energy - mpi) - mpi * mpi;
            if ((Q2max < Q2min) | (Q2min < 0))
                return 0.;

            // Set the binning
            const Scalar pQ2min = log(Q2min);
            const Scalar pQ2max = log(Q2max);
            const Scalar dpQ2 = pQ2max - pQ2min;
            const Scalar pQ2c = 0.5 * (pQ2max + pQ2min);

            /*
             * Integrate the doubly differential cross-section over Q2 using
             * a Gaussian quadrature. Note that 9 points are enough to get a
             * better than 0.1 % accuracy.
            */
            const auto ds =
                    utils::numerics::quadrature9<Scalar>(
                            0.f, 1.f,
                            [&A, &pQ2c, &dpQ2, &mass, &kinetic_energy, &recoil_energy](
                                    const Scalar &t) {
                                const Scalar Q2 = exp(pQ2c + 0.5 * dpQ2 * t);
                                return dcs_photonuclear_d2(A, mass, kinetic_energy,
                                                           recoil_energy, Q2) * Q2;
                            });

            return (ds < 0.) ? 0. : 0.5 * ds * dpQ2 * 1E+03 * AVOGADRO_NUMBER * (mass + kinetic_energy) / A;
        };


        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9620
         */
        inline const auto ionisation = [](const Energy &kinetic_energy,
                                          const Energy &recoil_energy,
                                          const AtomicElement<Scalar> &element,
                                          const ParticleMass &mass) {
            const Scalar A = element.A;
            const Index Z = element.Z;

            const Scalar P2 = kinetic_energy * (kinetic_energy + 2. * mass);
            const Scalar E = kinetic_energy + mass;
            const Scalar Wmax = 2. * ELECTRON_MASS * P2 /
                                (mass * mass +
                                 ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
            if ((Wmax < X_FRACTION * kinetic_energy) || (recoil_energy > Wmax))
                return (Scalar) 0.;
            const Scalar Wmin = 0.62 * element.I;
            if (recoil_energy <= Wmin)
                return (Scalar) 0.;

            // Close interactions for Q >> atomic binding energies
            const Scalar a0 = 0.5 / P2;
            const Scalar a1 = -1. / Wmax;
            const Scalar a2 = E * E / P2;
            const Scalar cs =
                    1.535336E-05 * E * Z / A * (a0 + 1. / recoil_energy * (a1 + a2 / recoil_energy));

            // Radiative correction
            Scalar Delta = 0.;
            const Scalar m1 = mass - ELECTRON_MASS;
            if (kinetic_energy >= 0.5 * m1 * m1 / ELECTRON_MASS) {
                const Scalar L1 = log(1. + 2. * recoil_energy / ELECTRON_MASS);
                Delta = 1.16141E-03 * L1 *
                        (log(4. * E * (E - recoil_energy) / (mass * mass)) -
                         L1);
            }
            return (Scalar) (cs * (1. + Delta));
        };

        // Close interactions for Q >> atomic binding energies.
        inline const auto analytic_del_ionisation_interactions = [](
                const Scalar &a0,
                const Scalar &a1,
                const Scalar &a2,
                const Scalar &Wmax,
                const Scalar &Wmin
        ) {
            return a0 * (Wmax - Wmin) + a1 * log(Wmax / Wmin) +
                   a2 * (1. / Wmin - 1. / Wmax);
        };

        inline const auto analytic_cel_ionisation_interactions = [](
                const Scalar &a0,
                const Scalar &a1,
                const Scalar &a2,
                const Scalar &Wmax,
                const Scalar &Wmin
        ) {
            return 0.5 * a0 * (Wmax * Wmax - Wmin * Wmin) +
                   a1 * (Wmax - Wmin) + a2 * log(Wmax / Wmin);
        };


        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9669
         */
        template<typename CloseInteractionsTerm>
        inline const auto analytic_recoil_energy_integral_ionisation =
                [](const Energy &kinetic_energy,
                   const EnergyTransfer &xlow,
                   const AtomicElement<Scalar> &element,
                   const ParticleMass &mass,
                   const CloseInteractionsTerm &interaction_term
                ) {
                    const Scalar P2 = kinetic_energy * (kinetic_energy + 2. * mass);
                    const Scalar E = kinetic_energy + mass;
                    const Scalar Wmax = 2. * ELECTRON_MASS * P2 /
                                        (mass * mass +
                                         ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
                    if (Wmax < X_FRACTION * kinetic_energy)
                        return (Scalar) 0.;
                    Scalar Wmin = 0.62 * element.I;
                    const Scalar qlow = kinetic_energy * xlow;
                    if (qlow >= Wmin)
                        Wmin = qlow;

                    // Check the bounds.
                    if (Wmax <= Wmin)
                        return (Scalar) 0.;

                    return (Scalar) (
                            1.535336E-05 * element.Z / element.A *
                            interaction_term(0.5 / P2, -1. / Wmax, E * E / P2, Wmax, Wmin));
                };

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6054
         */
        inline Scalar coulomb_frame_parameters(Scalar *fCM,
                                               const Energy &kinetic_energy,
                                               const AtomicElement<Scalar> &element,
                                               const ParticleMass &mass) {
            Scalar kinetic0;
            const Scalar Ma = element.A * ATOMIC_MASS_ENERGY;
            Scalar M2 = mass + Ma;
            M2 *= M2;
            const Scalar sCM12i = 1. / sqrt(M2 + 2. * Ma * kinetic_energy);
            fCM[0] = (kinetic_energy + mass + Ma) * sCM12i;
            kinetic0 =
                    (kinetic_energy * Ma + mass * (mass + Ma)) * sCM12i -
                    mass;
            if (kinetic0 < KIN_CUTOFF)
                kinetic0 = KIN_CUTOFF;
            const Scalar etot = kinetic_energy + mass + Ma;
            const Scalar betaCM2 =
                    kinetic_energy * (kinetic_energy + 2. * mass) / (etot * etot);
            Scalar rM2 = mass / Ma;
            rM2 *= rM2;
            fCM[1] = sqrt(rM2 * (1. - betaCM2) + betaCM2);
            return kinetic0;
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6038
         */
        inline Scalar coulomb_spin_factor(const Energy &kinetic_energy, const ParticleMass &mass) {
            const Scalar e = kinetic_energy + mass;
            return kinetic_energy * (e + mass) / (e * e);
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L5992
         */
        inline Scalar coulomb_wentzel_path(const Scalar &screening,
                                           const Energy &kinetic_energy,
                                           const AtomicElement<Scalar> &element,
                                           const ParticleMass &mass) {
            const Scalar d = kinetic_energy * (kinetic_energy + 2. * mass) /
                             (element.Z * (kinetic_energy + mass));
            return element.A * 2.54910918E+08 * screening * (1. + screening) * d * d;
        }


        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L5934
         */
        inline Scalar coulomb_screening_parameters(Scalar *pscreen,
                                                   const Energy &kinetic_energy,
                                                   const AtomicElement<Scalar> &element,
                                                   const ParticleMass &mass) {
            // Nuclear screening
            const Scalar third = 1. / 3;
            const Scalar A13 = pow(element.A, third);
            const Scalar R1 = 1.02934 * A13 + 0.435;
            const Scalar R2 = 2.;
            const Scalar p2 = kinetic_energy * (kinetic_energy + 2. * mass);
            const Scalar d = 5.8406E-02 / p2;
            pscreen[1] = d / (R1 * R1);
            pscreen[2] = d / (R2 * R2);

            // Atomic Moliere screening with Coulomb correction from Kuraev et al.
            // Phys. Rev. D 89, 116016 (2014). Valid for ultra-relativistic
            // particles only.

            const Index Z = element.Z;
            const Scalar etot = kinetic_energy + mass;
            const Scalar ZE = Z * etot;
            const Scalar zeta2 = 5.3251346E-05 * (ZE * ZE) / p2;
            Scalar cK;
            if (zeta2 > 1.) {
                // Let's perform the serie computation.
                Index i, n = 10 + Z;
                Scalar f = 0.;
                for (i = 1; i <= n; i++)
                    f += zeta2 / (i * (i * i + zeta2));
                cK = exp(f);
            } else {
                // Let's use Kuraev's approximate expression.
                cK = exp(1. - 1. / (1. + zeta2) +
                         zeta2 * (0.2021 + zeta2 * (0.0083 * zeta2 - 0.0369)));
            }

            // Original Moliere's atomic screening, considered as a reference
            // value at low energies.

            const Scalar cM = 1. + 3.34 * zeta2;

            // Atomic screening interpolation.
            Scalar r = kinetic_energy / etot;
            r *= r;
            const Scalar c = r * cK + (1. - r) * cM;
            pscreen[0] = 5.179587126E-12 * pow(Z, 2. / 3.) * c / p2;

            const Scalar d01 = 1. / (pscreen[0] - pscreen[1]);
            const Scalar d02 = 1. / (pscreen[0] - pscreen[2]);
            const Scalar d12 = 1. / (pscreen[1] - pscreen[2]);
            pscreen[6] = d01 * d01 * d02 * d02;
            pscreen[7] = d01 * d01 * d12 * d12;
            pscreen[8] = d12 * d12 * d02 * d02;
            pscreen[3] = 2. * pscreen[6] * (d01 + d02);
            pscreen[4] = 2. * pscreen[7] * (d12 - d01);
            pscreen[5] = -2. * pscreen[8] * (d12 + d02);

            return 1. / coulomb_wentzel_path(pscreen[0], kinetic_energy, element, mass);
        }

        inline const auto coulomb_data =
                [](
                        const CMLorentz &fCM,
                        const ScreeningFactors &screening,
                        const FSpins &fspin,
                        const InvLambdas &invlambda,
                        const Energies &kinetic_energies,
                        const AtomicElement<Scalar> &element,
                        const ParticleMass &mass) {
                    const Index nkin = kinetic_energies.numel();
                    auto *pK = kinetic_energies.data_ptr<Scalar>();

                    auto *pfCM = fCM.data_ptr<Scalar>();
                    auto *pscreen = screening.data_ptr<Scalar>();
                    auto *pfspin = fspin.data_ptr<Scalar>();
                    auto *pinvlbd = invlambda.data_ptr<Scalar>();

                    for (Index i = 0; i < nkin; i++) {
                        const Scalar kinetic0 = coulomb_frame_parameters(pfCM + 2 * i, pK[i], element, mass);
                        pfspin[i] = coulomb_spin_factor(kinetic0, mass);
                        pinvlbd[i] = coulomb_screening_parameters(pscreen + NSF * i, kinetic0, element, mass);
                    }
                };

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6160
         */
        inline void coulomb_transport_coefficients(
                Scalar *pcoefs,
                const Scalar *pscreen,
                const Scalar &fspin,
                const Scalar &mu) {
            const Scalar nuclear_screening =
                    (pscreen[1] < pscreen[2]) ? pscreen[1] : pscreen[2];
            if (mu < 1E-08 * nuclear_screening) {
                // We neglect the nucleus finite size.
                const Scalar L = log(1. + mu / pscreen[0]);
                const Scalar r = mu / (mu + pscreen[0]);
                const Scalar k = pscreen[0] * (1. + pscreen[0]);
                pcoefs[0] = k * (r / pscreen[0] - fspin * (L - r));
                const Scalar I2 = mu - pscreen[0] * (r - 2. * L);
                pcoefs[1] = 2. * k * (L - r - fspin * I2);
            } else {
                // We need to take all factors into account using a pole reduction.
                Scalar I0[3], I1[3], I2[3], J0[3], J1[3], J2[3];
                Index i;
                Scalar mu2 = 0.5 * mu * mu;
                for (i = 0; i < 3; i++) {
                    Scalar r = mu / (mu + pscreen[i]);
                    Scalar L = log(1. + mu / pscreen[i]);
                    Scalar mu1 = mu;
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

                const Scalar k = pscreen[0] * (1. + pscreen[0]) *
                                 pscreen[1] * pscreen[1] * pscreen[2] * pscreen[2];
                pcoefs[0] = pcoefs[1] = 0.;
                for (i = 0; i < 3; i++) {
                    pcoefs[0] += pscreen[3 + i] * (J0[i] - fspin * J1[i]) +
                                 pscreen[6 + i] * (I0[i] - fspin * I1[i]);
                    pcoefs[1] += pscreen[3 + i] * (J1[i] - fspin * J2[i]) +
                                 pscreen[6 + i] * (I1[i] - fspin * I2[i]);
                }
                pcoefs[0] *= k;
                pcoefs[1] *= 2. * k;
            }
        }

        inline const auto coulomb_transport =
                [](const TransportCoefs &coefficients,
                   const ScreeningFactors &screening,
                   const FSpins &fspin,
                   const AngularCutoff &mu) {
                    auto *pcoefs = coefficients.data_ptr<Scalar>();
                    auto *pscreen = screening.data_ptr<Scalar>();
                    auto *pfspin = fspin.data_ptr<Scalar>();

                    const bool nmu = (mu.numel() == 1);
                    auto *pmu = mu.data_ptr<Scalar>();

                    const Index nspin = fspin.numel();
                    for (Index i = 0; i < nspin; i++)
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
        inline Scalar coulomb_restricted_cs(
                const Scalar &mu,
                const Scalar &fspin,
                const Scalar *screen) {
            if (mu >= 1.)
                return 0.;

            const Scalar nuclear_screening =
                    (screen[1] < screen[2]) ? screen[1] : screen[2];
            if (mu < 1E-08 * nuclear_screening) {
                // We neglect the nucleus finite size.
                const Scalar L = log((screen[0] + 1.) / (screen[0] + mu));
                const Scalar r =
                        (1. - mu) / ((screen[0] + mu) * (screen[0] + 1.));
                const Scalar k = screen[0] * (1. + screen[0]);
                return k * (r - fspin * (L - screen[0] * r));
            } else {
                // We need to take all factors into account using a pole reduction.
                Scalar I0[3], I1[3], J0[3], J1[3];
                Index i;
                for (i = 0; i < 3; i++) {
                    const Scalar L =
                            log((screen[i] + 1.) / (screen[i] + mu));
                    const Scalar r = (1. - mu) /
                                     ((screen[i] + mu) * (screen[i] + 1.));
                    I0[i] = r;
                    J0[i] = L;
                    I1[i] = L - screen[i] * r;
                    J1[i] = mu - screen[i] * L;
                }

                const Scalar k = screen[0] * (1. + screen[0]) *
                                 screen[1] * screen[1] * screen[2] * screen[2];
                Scalar cs = 0.;
                for (i = 0; i < 3; i++) {
                    cs += screen[3 + i] * (J0[i] - fspin * J1[i]) +
                          screen[6 + i] * (I0[i] - fspin * I1[i]);
                }
                return k * cs;
            }
        }

        inline Scalar cutoff_objective(
                const Scalar &cs_h,
                const Scalar &mu,
                const Scalar *invlambda,
                Scalar *fspin,
                Scalar *screen,
                const Index nel = 1,
                const Index nkin = 1) {
            Scalar cs_tot = 0.;
            for (Index iel = 0; iel < nel; iel++) {
                const Index off = iel * nkin;
                cs_tot += invlambda[off] * coulomb_restricted_cs(mu, fspin[off], screen + NSF * off);
            }
            return cs_tot - cs_h;
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8472
         */
        inline void coulomb_hard_scattering(Scalar &mu0, Scalar &lb_h,
                                            const Scalar *G, const Scalar *fCM,
                                            Scalar *screen,
                                            Scalar *invlambda,
                                            Scalar *fspin,
                                            const Index nel = 1,
                                            const Index nkin = 1) {

            Scalar invlb_m = 0., invlb1_m = 0.;
            Scalar s_m_l = 0., s_m_h = 0.;

            for (Index iel = 0; iel < nel; iel++) {
                const Index off = iel * nkin;

                const Scalar invlb = invlambda[off];
                const Scalar scr = screen[NSF * off];

                invlb_m += invlb * G[2 * off];
                s_m_h += scr * invlb;
                s_m_l += invlb / scr;
                const Scalar d = 1. / (fCM[2 * off] * (1. + fCM[1 + 2 * off]));
                invlb1_m += invlb * G[1 + 2 * off] * d * d;
            }

            // Set the hard scattering mean free path.
            const Scalar lb_m = 1. / invlb_m;
            lb_h = std::min(EHS_OVER_MSC / invlb1_m, EHS_PATH_MAX);

            // Compute the hard scattering cutoff angle, in the CM.
            if (lb_m < lb_h) {
                // Initialise the root finder with an asymptotic starting value
                // Asymptotic value when lb_h >> lb_m versus when lb_h ~= lb_m
                Scalar s_m = (lb_h > 2. * lb_m) ? s_m_h * lb_m : 1. / (s_m_l * lb_m);

                mu0 = s_m * (lb_h - lb_m) / (s_m * lb_h + lb_m);

                // targeted cross section
                const Scalar cs_h = 1. / lb_h;

                // Configure for the root solver.
                // Solve for the cut-off angle. We try an initial bracketing in
                // [0.25*mu0; 4.*mu0], with mu0 the asymptotic estimate.
                Scalar mu_max = std::min(4. * mu0, 1.);
                Scalar mu_min = 0.25 * mu0;

                Scalar fmax = 0, fmin = 0;

                fmax = cutoff_objective(cs_h, mu_max, invlambda, fspin, screen, nel, nkin);
                if (fmax > 0.) {
                    // This shouldn't occur, but let's be safe and handle this case.
                    mu_min = mu_max;
                    fmin = fmax;
                    mu_max = 1.;
                    fmax = -cs_h;
                } else {
                    fmin = cutoff_objective(cs_h, mu_min, invlambda, fspin, screen, nel, nkin);
                    if (fmin < 0.) {
                        // This might occur at high energies when the nuclear screening becomes significant.
                        mu_max = mu_min;
                        fmax = fmin;
                        mu_min = 0.;
                        fmin = cutoff_objective(cs_h, mu_min, invlambda, fspin, screen, nel, nkin);
                    }
                    if (mu_min < MAX_MU0) {
                        mu_max = std::min(mu_max, MAX_MU0);
                        const auto mubest =
                                utils::numerics::ridders_root<Scalar>(
                                        mu_min, mu_max,
                                        [&](const Scalar &mu_x) {
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
            } else {
                lb_h = lb_m;
                mu0 = 0;
            }
        }


        inline const auto hard_scattering =
                [](const AngularCutoff &mu0,
                   const HSMeanFreePath &lb_h,
                   const TransportCoefs &coefficients,
                   const CMLorentz &transform,
                   const ScreeningFactors &screening,
                   const InvLambdas &invlambdas,
                   const FSpins &fspins) {
                    const Index nel = invlambdas.size(0);
                    const Index nkin = invlambdas.size(1);

                    auto *pmu0 = mu0.data_ptr<Scalar>();
                    auto *plb_h = lb_h.data_ptr<Scalar>();

                    auto *invlambda = invlambdas.data_ptr<Scalar>();
                    auto *fspin = fspins.data_ptr<Scalar>();
                    auto *G = coefficients.data_ptr<Scalar>();
                    auto *fCM = transform.data_ptr<Scalar>();
                    auto *screen = screening.data_ptr<Scalar>();

                    for (Index i = 0; i < nkin; i++)
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
        inline Scalar transverse_transport_ionisation(
                const Energy &kinetic_energy,
                const AtomicElement<Scalar> &element,
                const ParticleMass &mass) {
            // Soft close interactions, restricted to X_FRACTION.
            const Scalar momentum2 = kinetic_energy * (kinetic_energy + 2. * mass);
            const Scalar E = kinetic_energy + mass;
            const Scalar Wmax = 2. * ELECTRON_MASS * momentum2 /
                                (mass * mass +
                                 ELECTRON_MASS * (ELECTRON_MASS + 2. * E));
            const Scalar W0 = 2. * momentum2 / ELECTRON_MASS;
            const Scalar mu_max = Wmax / W0;
            Scalar mu3 = kinetic_energy * X_FRACTION / W0;
            if (mu3 > mu_max)
                mu3 = mu_max;
            const Scalar mu2 = 0.62 * element.I / W0;
            if (mu2 >= mu3)
                return 0.;
            const Scalar a0 = 0.5 * W0 / momentum2;
            const Scalar a1 = -1. / Wmax;
            const Scalar a2 = E * E / (W0 * momentum2);
            const Scalar cs0 = 1.535336E-05 / element.A; /* m^2/kg/GeV. */
            return 2. * cs0 * element.Z *
                   (0.5 * a0 * (mu3 * mu3 - mu2 * mu2) + a1 * (mu3 - mu2) +
                    a2 * log(mu3 / mu2));
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L6262
         */
        inline Scalar transverse_transport_photonuclear(
                const Energy &kinetic_energy,
                const AtomicElement<Scalar> &element,
                const ParticleMass &mass) {
            // Integration over the kinetic_energy transfer, q, done with a log sampling.
            const Scalar E = kinetic_energy + mass;
            return 2. * utils::numerics::quadrature6<Scalar>(
                    log(1E-06), 0.,
                    [&](const Scalar &t) {
                        const Scalar nu = X_FRACTION * exp(t);
                        const Scalar q = nu * kinetic_energy;

                        // Analytical integration over mu.
                        const Scalar m02 = 0.4;
                        const Scalar q2 = q * q;
                        const Scalar tmax = 1.876544 * q;
                        const Scalar tmin =
                                q2 * mass * mass / (E * (E - q));
                        const Scalar b1 = 1. / (1. - q2 / m02);
                        const Scalar c1 = 1. / (1. - m02 / q2);
                        Scalar L1 = b1 * log((q2 + tmax) / (q2 + tmin));
                        Scalar L2 = c1 * log((m02 + tmax) / (m02 + tmin));
                        const Scalar I0 = log(tmax / tmin) - L1 - L2;
                        L1 *= q2;
                        L2 *= m02;
                        const Scalar I1 = L1 + L2;
                        L1 *= q2;
                        L2 *= m02;
                        const Scalar I2 =
                                (tmax - tmin) * (b1 * q2 + c1 * m02) - L1 - L2;
                        const Scalar ratio =
                                (I1 * tmax - I2) / ((I0 * tmax - I1) * kinetic_energy *
                                                    (kinetic_energy + 2. * mass));

                        return photonuclear(kinetic_energy, q, element, mass) * ratio * nu;
                    },
                    100);
        }

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L8730
         */
        inline const auto soft_scattering =
                [](const Calculation &ms1,
                   const Energies &K,
                   const AtomicElement<Scalar> &element,
                   const ParticleMass &mass) {
                    utils::vmap<Scalar>(
                            K,
                            [&](const Scalar &k) { return transverse_transport_ionisation(k, element, mass) +
                                                          transverse_transport_photonuclear(k, element, mass); },
                            ms1);
                };

    } // namespace noa::pms::dcs::pumas

    template<>
    inline auto cel_integral<Scalar>(const decltype(pumas::ionisation) &dcs_func) {
        return [&dcs_func](const Scalar &kinetic_energy,
                           const Scalar &xlow,
                           const AtomicElement<Scalar> &element,
                           const Scalar &mass,
                           const Index min_points) {
            const Scalar m1 = mass - ELECTRON_MASS;
            return (kinetic_energy <= 0.5 * m1 * m1 / ELECTRON_MASS) ?
                   pumas::analytic_recoil_energy_integral_ionisation<decltype(pumas::analytic_cel_ionisation_interactions)>
                           (kinetic_energy, xlow, element, mass, pumas::analytic_cel_ionisation_interactions) :
                   recoil_energy_integral<Scalar>(
                           [&dcs_func](const Scalar &k,
                                       const Scalar &q,
                                       const AtomicElement<Scalar> &el,
                                       const ParticleMass &m) {
                               return dcs_func(k, q, el, m) * q * q;
                           })(kinetic_energy, xlow, element, mass, min_points);
        };
    }

    template<>
    inline auto del_integral<Scalar>(const decltype(pumas::ionisation) &dcs_func) {
        return [&dcs_func](const Scalar &kinetic_energy,
                           const Scalar &xlow,
                           const AtomicElement<Scalar> &element,
                           const Scalar &mass,
                           const Index min_points) {
            const Scalar m1 = mass - ELECTRON_MASS;
            return (kinetic_energy <= 0.5 * m1 * m1 / ELECTRON_MASS) ?
                   pumas::analytic_recoil_energy_integral_ionisation<decltype(pumas::analytic_del_ionisation_interactions)>
                           (kinetic_energy, xlow, element, mass, pumas::analytic_del_ionisation_interactions) :
                   recoil_energy_integral<Scalar>(
                           [&dcs_func](const Scalar &k,
                                       const Scalar &q,
                                       const AtomicElement<Scalar> &el,
                                       const ParticleMass &m) {
                               return dcs_func(k, q, el, m) * q;
                           })(kinetic_energy, xlow, element, mass, min_points);
        };
    }

} // namespace noa::pms::dcs