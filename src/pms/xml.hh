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

#include "pugixml.hpp"

namespace noa::pms::mdf
{
    using Document = pugi::xml_document;
    using Node = pugi::xml_node;


    template <typename MDFComponents, typename Component>
    inline std::optional<MDFComponents> get_mdf_components(
        const Node &node,
        const std::unordered_map<std::string, Component> &comp_data,
        const std::string &tag)
    {
        auto comp_xnodes = node.select_nodes("component[@name][@fraction]");
        auto comps = MDFComponents{};
        Scalar tot = 0.0;
        for (const auto &xnode : comp_xnodes)
        {
            auto node = xnode.node();
            auto name = node.attribute("name").value();
            Scalar frac = node.attribute("fraction").as_double();
            tot += frac;
            if (!comp_data.count(name))
            {
                std::cerr << "Cannot find component " << name << " for "
                          << tag << std::endl;
                return std::nullopt;
            }
            comps[name] = frac;
        }
        if (tot > 1.0)
        {
            std::cerr << "Fractions add up to " << tot << " for " << tag
                      << std::endl;
            return std::nullopt;
        }
        return comps;
    }



} // namespace noa::pms