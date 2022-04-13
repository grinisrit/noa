// Greg put your code there

#include <gflags/gflags.h>
#include <iostream>

DEFINE_string(mesh, "mesh.vtu", "Path to tetrahedron mesh");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for the mass lumping technique in MHFEM");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << FLAGS_mesh << std::endl;

    gflags::ShutDownCommandLineFlags();

    return 0;

}