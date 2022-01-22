#pragma once

#include <noa/utils/common.hh>

#include <noa/3rdparty/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Readers/VTUReader.h>
#include <noa/3rdparty/TNL/Meshes/Writers/VTUWriter.h>

using namespace noaTNL;
using namespace noaTNL::Meshes;
using TetrahedronMesh = Mesh<DefaultConfig<Topologies::Tetrahedron>>;

auto load_mesh(const noa::utils::Path &mesh_path) -> TetrahedronMesh {
    auto mesh = TetrahedronMesh{};
    auto reader = Readers::VTUReader{mesh_path};
    reader.loadMesh(mesh);

    std::cout << mesh.getMeshDimension() << "\n"
              << mesh.getEntitiesCount<mesh.getMeshDimension()>() << "\n";

    return mesh;
}