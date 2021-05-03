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
#include "noa/pms/mdf.hh"

namespace noa::pms {

    using ElementId = Index;
    using ElementIds = std::unordered_map<mdf::ElementName, ElementId>;
    using ElementNames = std::vector<mdf::ElementName>;

    using ElementIdsList = torch::Tensor;
    using ElementsFractions = torch::Tensor;

    template<typename Dtype>
    struct Material {
        ElementIdsList element_ids;
        ElementsFractions fractions;
        Dtype density;
    };

    using MaterialId = Index;
    using MaterialIds = std::unordered_map<mdf::MaterialName, MaterialId>;
    using MaterialNames = std::vector<mdf::MaterialName>;



    template<typename Dtype, typename Physics>
    class Model {
        using Elements = std::vector<AtomicElement<Dtype>>;
        using Materials = std::vector<Material<Dtype>>;

        Elements elements;
        ElementIds element_id;
        ElementNames element_name;

        Materials materials;
        MaterialIds material_id;
        MaterialNames material_name;

        // TODO: Composite materials

        inline Dtype energy_scale_MeV() const {
            return static_cast<Physics *>(this)->energy_scale_MeV_();
        }

        inline void set_elements(const mdf::Elements &mdf_elements) {
            int id = 0;
            elements.reserve(mdf_elements.size());
            for (auto[name, element] : mdf_elements) {
                element.I = 1E-6 * energy_scale_MeV() * element.I;
                elements.push_back(element);
                element_id[name] = id;
                element_name.push_back(name);
                id++;
            }
        }

    public:
        inline const AtomicElement<Dtype> &get_element(const ElementId id) const {
            return elements.at(id);
        }

        inline const AtomicElement<Dtype> &get_element(const mdf::ElementName &name) const {
            return elements.at(element_id.at(name));
        }

        inline const mdf::ElementName &get_element_name(const ElementId id) const {
            return element_name.at(id);
        }

        inline int num_elements() const {
            return elements.size();
        }

        inline const Material<Dtype> &get_material(const MaterialId id) const {
            return materials.at(id);
        }

        inline const Material<Dtype> &get_material(const mdf::MaterialName &name) const {
            return materials.at(material_id.at(name));
        }

        inline const mdf::MaterialName &get_material_name(const MaterialId id) const {
            return material_name.at(id);
        }

        inline int num_materials() const {
            return materials.size();
        }

        inline Model &set_mdf_settings(const mdf::Settings &mdf_settings){
            set_elements(std::get<mdf::Elements>(mdf_settings));
            return *this;
        }

    };

} //namespace noa::pms