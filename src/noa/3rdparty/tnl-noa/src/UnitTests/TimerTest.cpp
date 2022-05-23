// Implemented by Nina Dzugasova

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/Timer.h>

using namespace TNL;

#ifdef HAVE_GTEST
TEST( TimerTest, Constructor )
{
    Timer time;
    time.reset();
    EXPECT_EQ(time.getRealTime(),0);
    /*time.start();
    EXPECT_FALSE(time.stopState);

    time.stop();
    EXPECT_TRUE(time.stopState);

    EXPECT_NE(time.getRealTime(),0);*/
}
#endif

#include "main.h"
