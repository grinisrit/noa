#include <noa/utils/meshes.hh>
#include <noa/3rdparty/pumas.hh>

#include <gflags/gflags.h>

using namespace noa;

DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");
DEFINE_string(mesh, "noa-test-data/meshes/mesh.vtu", "Path to tetrahedron mesh");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto meshOpt = utils::meshes::load_tetrahedron_mesh(FLAGS_mesh);
    if (!meshOpt.has_value()){
        return 1;
    }
    auto& mesh = meshOpt.value();
    std::cout << mesh.getMeshDimension() << "\n";

    std::cout << noa::pumas::PUMAS_MODE_BACKWARD << std::endl;

    gflags::ShutDownCommandLineFlags();

    return 0;
}