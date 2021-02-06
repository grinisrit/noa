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

namespace ghmc::pms
{
    using ParticleMass = float;
    using DecayLength = float;
    using AtomicMass = float;
    using MeanExcitation = float;
    using AtomicNumber = int;
    struct AtomicElement
    {
        AtomicMass A;
        MeanExcitation I;
        AtomicNumber Z;
    };
    using MaterialDensity = float;

    inline const float AVOGADRO_NUMBER = 6.02214076E+23f;
    inline const ParticleMass ELECTRON_MASS = 0.510998910E-03f; // GeV/c^2
    inline const ParticleMass MUON_MASS = 0.10565839f;          // GeV/c^2
    inline const DecayLength MUON_CTAU = 658.654f;              // m
    
    // Relative switch between Continuous Energy Loss (CEL) and DELs.
    inline const float X_FRACTION = 5E-02f;

    inline const auto STANDARD_ROCK = AtomicElement{
        22.0f,      // g/mol
        0.1364E-6f, // GeV
        11};

} // namespace ghmc::pms