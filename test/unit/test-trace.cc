#include <noa/3rdparty/TNL/Assert.h>

#include <noa/utils/meshes.hh>

#include "test-trace.hh"

using namespace noa::utils::meshes;

HostMeshOpt load_tetrahedron_host_test_mesh() {
    return load_tetrahedron_mesh("tmesh.vtu");
}

TEST(TRACE, TraceFirstBorderInTetrahedron) {
    test_get_first_border_in_tetrahedron<Devices::Host>();
}

TEST(TRACE, CheckCurrentTetrahedron) {
    test_get_current_tetrahedron<Devices::Host>();
}