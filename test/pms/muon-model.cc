#include <noa/3rdparty/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Readers/VTUReader.h>
#include <noa/3rdparty/TNL/Meshes/Writers/VTUWriter.h>

using namespace noaTNL;
using namespace noaTNL::Meshes;
using TetrahedronMesh = Mesh<DefaultConfig<Topologies::Tetrahedron>>;


#include <gflags/gflags.h>

DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");
DEFINE_string(mesh, "noa-test-data/meshes/tmesh.vtu", "Path to tetrahedron mesh");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto mesh = TetrahedronMesh{};
    auto reader = Readers::VTUReader{FLAGS_mesh};
    reader.loadMesh(mesh);

    std::cout << mesh.getMeshDimension() << "\n"
              << mesh.getEntitiesCount<mesh.getMeshDimension()>() << "\n";

    gflags::ShutDownCommandLineFlags();

    return 0;
}