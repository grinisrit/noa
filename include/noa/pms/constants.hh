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

#include <torch/types.h>

namespace noa::pms {
    using Scalar = double;
    using Index = int;
    using UniversalConst = Scalar;
    using ParticleMass = Scalar;
    using DecayLength = Scalar;
    using AtomicNumber = Index;

    template<typename Dtype>
    struct AtomicElement {
        Dtype A; // Atomic mass
        Dtype I; // Mean Excitation
        AtomicNumber Z;
    };

    using MaterialDensity = Scalar;
    using Energy = Scalar;
    using EnergyTransfer = Scalar; // Relative energy transfer
    using ComponentFraction = Scalar;
    using LarmorFactor = Scalar;

    template<typename Dtype>
    struct MaterialDensityEffect {
        Dtype a, k, x0, x1, Cbar, delta0; // Sternheimer coefficients
    };

    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using DCSCalculation = torch::Tensor; // Receiver tensor for DCS calculations
    using Tabulation = torch::Tensor;     // Tabulations holder

    constexpr UniversalConst AVOGADRO_NUMBER = 6.02214076E+23;
    constexpr UniversalConst ATOMIC_MASS_ENERGY = 0.931494; // Atomic mass to MeV

    constexpr ParticleMass ELECTRON_MASS = 0.510998910E-03; // GeV/c^2
    constexpr ParticleMass MUON_MASS = 0.10565839;          // GeV/c^2
    constexpr ParticleMass TAU_MASS = 1.77682;              // GeV/c^2

    constexpr DecayLength MUON_CTAU = 658.654;  // m
    constexpr DecayLength TAU_CTAU = 87.03E-06; // m

    constexpr LarmorFactor LARMOR_FACTOR = 0.299792458; // m^-1 GeV/c T^-1.

    // Common elements:
    constexpr AtomicElement<Scalar> STANDARD_ROCK =
            AtomicElement<Scalar>{
                    22.,       // g/mol
                    0.1364E-6, // GeV
                    11};

    // Default relative switch between continuous and discrete energy loss
    constexpr EnergyTransfer X_FRACTION = 5E-02;
    // Maximum allowed energy transfer for using DCS models
    constexpr EnergyTransfer DCS_MODEL_MAX_FRACTION = 0.95;

    constexpr Energy KIN_CUTOFF = 1E-9;           // GeV, energy cutoff used in relativistic kinematics
    constexpr Scalar EHS_PATH_MAX = 1E+9;         // kg/m^2, max inverse path length for Elastic Hard Scattering (EHS) events
    constexpr Scalar EHS_OVER_MSC = 1E-4;         // EHS to 1st transport multiple scattering interaction length path ratio
    constexpr Scalar MAX_SOFT_ANGLE = 1E+00;      // degrees, max deflection angle for a soft scattering event
    constexpr Energy DCS_MODEL_MIN_KINETIC = 10.; // GeV, Minimum kinetic energy for using the DCS model

    inline const Scalar MAX_MU0 =
            0.5 * (1. - cos(MAX_SOFT_ANGLE * M_PI / 180.)); // Max deflection angle for hard scattering

    namespace pumas {

        constexpr Scalar ENERGY_SCALE = 1E-3; // from MeV to GeV
        constexpr Scalar DENSITY_SCALE = 1E+3; // from g/cm^3 to kg/m^3

        // Default relative switch between continuous and discrete energy loss
        constexpr EnergyTransfer X_FRACTION = 5E-02;
        // Maximum allowed energy transfer for using DCS models
        constexpr EnergyTransfer DCS_MODEL_MAX_FRACTION = 0.95;

        constexpr Energy KIN_CUTOFF = 1E-9;           // GeV, energy cutoff used in relativistic kinematics
        constexpr Scalar EHS_PATH_MAX = 1E+9;         // kg/m^2, max inverse path length for Elastic Hard Scattering (EHS) events
        constexpr Scalar EHS_OVER_MSC = 1E-4;         // EHS to 1st transport multiple scattering interaction length path ratio
        constexpr Scalar MAX_SOFT_ANGLE = 1E+00;      // degrees, max deflection angle for a soft scattering event
        constexpr Energy DCS_MODEL_MIN_KINETIC = 10.; // GeV, Minimum kinetic energy for using the DCS model

        inline const Scalar MAX_MU0 =
                0.5 * (1. - cos(MAX_SOFT_ANGLE * M_PI / 180.)); // Max deflection angle for hard scattering

        constexpr int NPR = 4;  // Number of DEL processes considered
        constexpr int NSF = 9;  // Number of screening factors and pole reduction for Coulomb scattering
        constexpr int NLAR = 8; // Order of expansion for the computation of the magnetic deflection

        // Polynomials order for the DCS model
        constexpr int DCS_MODEL_ORDER_P = 6;
        constexpr int DCS_MODEL_ORDER_Q = 2;
        constexpr int DCS_SAMPLING_N = 11; // Samples for DCS model
        constexpr int NDM = DCS_MODEL_ORDER_P + DCS_MODEL_ORDER_Q + DCS_SAMPLING_N + 1;

        using ComputeCEL = bool;      // Compute Continuous Energy Loss (CEL) flag
        using MomentumIntegral = Scalar;

        using ThresholdIndex = Index;
        using Thresholds = torch::Tensor;

        using InvLambdas = torch::Tensor;       // Inverse of the mean free grammage
        using ScreeningFactors = torch::Tensor; // Atomic and nuclear screening factors & pole reduction
        using FSpins = torch::Tensor;           // Spin corrections
        using CMLorentz = torch::Tensor;        // Center of Mass to Observer frame transform
        using AngularCutoff = torch::Tensor;    // Cutoff angle for coulomb scattering
        using HSMeanFreePath = torch::Tensor;   // Hard scattering mean free path
        using TransportCoefs = torch::Tensor;
        using SoftScatter = torch::Tensor; // Soft scattering terms per element
    }

} // namespace noa::pms