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

#include "noa/pms/mdf.hh"

namespace noa::pms {

    using KineticEnergies = torch::Tensor;
    using RecoilEnergies = torch::Tensor;
    using Table = torch::Tensor;  // Generic table
    using Result = torch::Tensor; // Receiver tensor for calculations result

    using ElementId = Index;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ELementIdsList = torch::Tensor;
    using ElementsFractions = torch::Tensor;
    using ElementNames = std::vector<mdf::ElementName>;

    struct Material {
        ELementIdsList element_ids;
        ElementsFractions fractions;
    };
    using Materials = std::vector<Material>;
    using MaterialId = Index;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using MaterialNames = std::vector<mdf::MaterialName>;

    using MaterialDensities = torch::Tensor;
    using MaterialRelativeElectronicDensities = torch::Tensor;
    using MaterialMeanExcitationEnergies = torch::Tensor;

    template<typename Dtype, typename Physics>
    class Model {
        using Elements = std::vector<AtomicElement<Dtype>>;

    };

} //namespace noa::pms