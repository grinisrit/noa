#include "noa-test.hh"

#include <gtest/gtest.h>


int main(int argc, char *argv[])
{

    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new TestDataEnv<GHMCData>);

    ::testing::AddGlobalTestEnvironment(new TestDataEnv<DCSData>);

    return RUN_ALL_TESTS();
}