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

#include <torch/torch.h>

namespace ghmc::pms
{
    using Scalar = double;
    using Index = int;
    using Index64 = int64_t;
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
    using EnergyTransferMin = Scalar; // Relative lower bound of energy transfer
    using ComponentFraction = Scalar;

    struct MaterialDensityEffect
    {
        Scalar a, k, x0, x1, Cbar, delta0; // Sternheimer coefficients
    };

    constexpr UniversalConst AVOGADRO_NUMBER = 6.02214076E+23;
    constexpr UniversalConst ATOM_ENERGY = 931.494; // Atomic unit to MeV

    constexpr ParticleMass ELECTRON_MASS = 0.510998910E-03; // GeV/c^2
    constexpr ParticleMass MUON_MASS = 0.10565839;          // GeV/c^2
    constexpr ParticleMass TAU_MASS = 1.77682;              // GeV/c^2

    constexpr DecayLength MUON_CTAU = 658.654;  // m
    constexpr DecayLength TAU_CTAU = 87.03E-06; // m

    // Relative switch between Continuous Energy Loss (CEL) and DELs.
    constexpr EnergyTransferMin X_FRACTION = 5E-02;

    // Common elements:
    constexpr AtomicElement STANDARD_ROCK = AtomicElement{
        22.,       // g/mol
        0.1364E-6, // GeV
        11};

    constexpr Scalar KIN_CUTOFF = 1E-6; // MeV, used in relativistic kinematics

} // namespace ghmc::pms