/**
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

#include <torch/types.h>

namespace noa::pms {
    using Scalar = double_t;
    using Index = int32_t;
    using UniversalConst = Scalar;
    using ParticleMass = Scalar;
    using DecayLength = Scalar;
    using AtomicNumber = Index;
    using AtomicMass = Scalar;
    using MeanExcitation = Scalar;
    using MaterialDensity = Scalar;

    struct AtomicElement {
        AtomicMass A;
        MeanExcitation I;
        AtomicNumber Z;
    };


    using Energy = Scalar;
    using EnergyTransfer = Scalar; // Relative energy transfer
    using LarmorFactor = Scalar;

    using Energies = torch::Tensor;
    using Calculation = torch::Tensor; // Receiver tensor for calculations
    using Tabulation = torch::Tensor;  // Tabulations holder

    constexpr UniversalConst AVOGADRO_NUMBER = 6.02214076E+23;
    constexpr UniversalConst ATOMIC_MASS_ENERGY = 0.931494; // Atomic mass to MeV

    constexpr ParticleMass ELECTRON_MASS = 0.510998910E-03; // GeV/c^2
    constexpr ParticleMass MUON_MASS = 0.10565839;          // GeV/c^2
    constexpr ParticleMass TAU_MASS = 1.77682;              // GeV/c^2

    constexpr DecayLength MUON_CTAU = 658.654;  // m
    constexpr DecayLength TAU_CTAU = 87.03E-06; // m

    constexpr LarmorFactor LARMOR_FACTOR = 0.299792458; // m^-1 GeV/c T^-1.

    // Common elements:
    constexpr AtomicElement STANDARD_ROCK =
            AtomicElement{
                    22.,       // g/mol
                    0.1364E-6, // GeV
                    11};

    namespace dcs {

        // Default relative switch between continuous and discrete energy loss
        constexpr EnergyTransfer X_FRACTION = 5E-02;

        constexpr Energy KIN_CUTOFF = 1E-9;           // GeV, energy cutoff used in relativistic kinematics
        constexpr Scalar EHS_PATH_MAX = 1E+9;         // kg/m^2, max inverse path length for Elastic Hard Scattering (EHS) events
        constexpr Scalar EHS_OVER_MSC = 1E-4;         // EHS to 1st transport multiple scattering interaction length path ratio
        constexpr Scalar MAX_SOFT_ANGLE = 1E+00;      // degrees, max deflection angle for a soft scattering event

        inline const Scalar MAX_MU0 =
                0.5 * (1. - cos(MAX_SOFT_ANGLE * M_PI / 180.)); // Max deflection angle for hard scattering

        constexpr Index NPR = 4;  // Number of DEL processes considered
        constexpr Index NSF = 9;  // Number of screening factors and pole reduction for Coulomb scattering
        constexpr Index NLAR = 8; // Order of expansion for the computation of the magnetic deflection

        // Polynomials order for the DCS model
        constexpr Index DCS_MODEL_ORDER_P = 6;
        constexpr Index DCS_MODEL_ORDER_Q = 2;
        constexpr Index DCS_SAMPLING_N = 11; // Samples for DCS model
        constexpr Index NDM = DCS_MODEL_ORDER_P + DCS_MODEL_ORDER_Q + DCS_SAMPLING_N + 1;

        using MomentumIntegral = Scalar;

        using InvLambdas = torch::Tensor;       // Inverse of the mean free grammage
        using ScreeningFactors = torch::Tensor; // Atomic and nuclear screening factors & pole reduction
        using FSpins = torch::Tensor;           // Spin corrections
        using CMLorentz = torch::Tensor;        // Center of Mass to Observer frame transform
        using AngularCutoff = torch::Tensor;    // Cutoff angle for coulomb scattering
        using HSMeanFreePath = torch::Tensor;   // Hard scattering mean free path
        using TransportCoefs = torch::Tensor;
        using SoftScatter = torch::Tensor; // Soft scattering terms per element


#ifdef __NVCC__
        __device__ __forceinline__
#else

        inline
#endif
        Scalar _bremsstrahlung_(
                const Energy &kinetic_energy,
                const Energy &recoil_energy,
                const AtomicElement &element,
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
            const Scalar dcs_factor = 7.297182E-07 * rem * rem * Z;

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
            return (dcs < 0.) ? 0. : dcs * 1E+03 * AVOGADRO_NUMBER / A;
        }

    } // namespace noa::pms::dcs

} // namespace noa::pms