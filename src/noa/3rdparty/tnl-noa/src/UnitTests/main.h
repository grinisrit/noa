#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include "GtestPrintToOverrides.h"
#else
#include "GtestMissingError.h"
#endif

int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
