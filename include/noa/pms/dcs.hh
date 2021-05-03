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

    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using DCSCalculation = torch::Tensor; // Receiver tensor for DCS calculations

    template<typename Dtype, typename DCSFunc>
    inline auto vmap(const DCSFunc &dcs_func) {
        return [&dcs_func](const DCSCalculation &result,
                           const KineticEnergies &kinetic_energies,
                           const RecoilEnergies &recoil_energies,
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
        return [&dcs_func](const DCSCalculation &result,
                           const KineticEnergies &kinetic_energies,
                           const RecoilEnergies &recoil_energies,
                           const AtomicElement<Dtype> &element,
                           const Dtype &mass) {
            const Dtype *recoil_energy = recoil_energies.data_ptr<Dtype>();
            utils::pvmapi<Dtype>(
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
                const Energy &kinetic_energy,
                const Energy &recoil_energy,
                const AtomicElement<Scalar> &element,
                const ParticleMass &mass) {
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

        /*
         *  Following closely the implementation by Valentin NIESS (niess@in2p3.fr)
         *  GNU Lesser General Public License version 3
         *  https://github.com/niess/pumas/blob/d04dce6388bc0928e7bd6912d5b364df4afa1089/src/pumas.c#L9221
         */
        inline const auto pair_production = [](const Energy &kinetic_energy,
                                               const Energy &recoil_energy,
                                               const AtomicElement<Scalar> &element,
                                               const ParticleMass &mass) {
            const int Z = element.Z;
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
            Scalar I = utils::numerics::quadrature8<Scalar>(0.f, 1.f, [&](const Scalar &t) {
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
            const Scalar ds =
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
            const int Z = element.Z;

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

    } // namespace noa::pms::dcs::pumas
} // namespace noa::pms::dcs