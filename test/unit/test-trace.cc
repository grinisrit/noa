#include <noa/utils/meshes.hh>

#include "test-trace.hh"

using namespace noa::utils::meshes;

TEST(TRACE, TraceFirstBorderInTetrahedron) {
    test_get_first_border_in_tetrahedron<Devices::Host>();
}

TEST(TRACE, CheckCurrentTetrahedron) {
    test_get_current_tetrahedron<Devices::Host>();
}

TEST(TRACE, CheckSideCases) {
    check_side_cases<Devices::Host>();
}