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

#include <noa/utils/common.hh>

#include <noa/3rdparty/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Readers/VTUReader.h>


namespace noa::utils::meshes {

    using namespace TNL;
    using TetrahedronMesh = Meshes::Mesh<Meshes::DefaultConfig<Meshes::Topologies::Tetrahedron>>;
    using TetrahedronMeshOpt = std::optional<TetrahedronMesh>;

    inline TetrahedronMeshOpt  load_tetrahedron_mesh(const Path& path) {
        if (check_path_exists(path)) {
            auto mesh = TetrahedronMesh{};
            auto reader = Meshes::Readers::VTUReader{path};
            try {
                reader.loadMesh(mesh);
            }
            catch (...) {
                std::cerr << "Failed to load tetrahedron mesh from " << path << "\n";
                return std::nullopt;
            }
            return std::make_optional(mesh);
        } else {
            return std::nullopt;
        }
    }

} // namespace noa::utils::meshes