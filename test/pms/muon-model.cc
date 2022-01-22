#include "load-mesh.hh"

#include <noa/3rdparty/pumas.hh>

#include <gflags/gflags.h>

DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");
DEFINE_string(mesh, "noa-test-data/meshes/tmesh.vtu", "Path to tetrahedron mesh");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    load_mesh(FLAGS_mesh);

    std::cout << noa::pumas::PUMAS_MODE_BACKWARD << std::endl;

    gflags::ShutDownCommandLineFlags();

    return 0;
}