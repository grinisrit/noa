#include "test-ghmc.hh"
#include "test-dcs.hh"

#include <gtest/gtest.h>

int main(int argc, char *argv[])
{

    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new GHMCData);

    ::testing::AddGlobalTestEnvironment(new DCSData);

    return RUN_ALL_TESTS();
}