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

#include "noa/pms/conf.hh"

namespace noa::pms
{
    using UniversalConst = Scalar;
    using ParticleMass = Scalar;
    using DecayLength = Scalar;
    using AtomicMass = Scalar;
    using MeanExcitation = Scalar;
    using AtomicNumber = Index;
    struct AtomicElement
    {
        AtomicMass A;
        MeanExcitation I;
        AtomicNumber Z;
    };
    using MaterialDensity = Scalar;
    using Energy = Scalar;
    using EnergyTransfer = Scalar; // Relative energy transfer
    using ComponentFraction = Scalar;
    using LarmorFactor = Scalar;

    struct MaterialDensityEffect
    {
        Scalar a, k, x0, x1, Cbar, delta0; // Sternheimer coefficients
    };

    constexpr UniversalConst AVOGADRO_NUMBER = 6.02214076E+23;
    constexpr UniversalConst ATOMIC_MASS_ENERGY = 0.931494; // Atomic mass to MeV

    constexpr ParticleMass ELECTRON_MASS = 0.510998910E-03; // GeV/c^2
    constexpr ParticleMass MUON_MASS = 0.10565839;          // GeV/c^2
    constexpr ParticleMass TAU_MASS = 1.77682;              // GeV/c^2

    constexpr DecayLength MUON_CTAU = 658.654;  // m
    constexpr DecayLength TAU_CTAU = 87.03E-06; // m

    constexpr LarmorFactor LARMOR_FACTOR = 0.299792458; // m^-1 GeV/c T^-1.

    // Default relative switch between continuous and discrete energy loss
    constexpr EnergyTransfer X_FRACTION = 5E-02;
    // Maximum allowed energy transfer for using DCS models
    constexpr EnergyTransfer DCS_MODEL_MAX_FRACTION = 0.95;

    // Common elements:
    constexpr AtomicElement STANDARD_ROCK = AtomicElement{
        22.,       // g/mol
        0.1364E-6, // GeV
        11};

    constexpr Energy KIN_CUTOFF = 1E-9;           // GeV, energy cutoff used in relativistic kinematics
    constexpr Scalar EHS_PATH_MAX = 1E+9;         // kg/m^2, max inverse path length for Elastic Hard Scattering (EHS) events
    constexpr Scalar EHS_OVER_MSC = 1E-4;         // EHS to 1st transport multiple scattering interaction length path ratio
    constexpr Scalar MAX_SOFT_ANGLE = 1E+00;      // degrees, max deflection angle for a soft scattering event
    constexpr Energy DCS_MODEL_MIN_KINETIC = 10.; // GeV, Minimum kinetic energy for using the DCS model

    inline const Scalar MAX_MU0 = 0.5 * (1. - cos(MAX_SOFT_ANGLE * M_PI / 180.)); // Max deflection angle for hard scattering

} // namespace noa::pms