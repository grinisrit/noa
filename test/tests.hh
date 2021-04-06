#pragma once

#include "test-data.hh"

#include <gtest/gtest.h>


template<typename TestData>
struct TestDataEnv: ::testing::Environment
{
     virtual void SetUp()
     {
        TestData::get_all();
     }
};