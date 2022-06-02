/*****************************************************************************
 *   Copyright (c) 2022, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Roland Grinis
 */

#pragma once

#include <noa/utils/common.hh>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/VTUReader.h>


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
            catch (const std::exception &exc) {
                std::cerr << "Failed to load tetrahedron mesh from " << path << "\n" << exc.what() << "\n";
                return std::nullopt;
            }
            return std::make_optional(mesh);
        } else {
            return std::nullopt;
        }
    }

} // namespace noa::utils::meshes
