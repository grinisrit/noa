#define HAVE_CUDA

#include <gtest/gtest.h>

#include "test-trace.hh"

TEST(TRACE, TraceFirstBorderInTetrahedronCuda) {
    test_get_first_border_in_tetrahedron<Devices::Cuda>();
}