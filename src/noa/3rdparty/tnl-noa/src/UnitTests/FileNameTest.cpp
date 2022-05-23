// Implemented by Nina Dzugasova

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/FileName.h>
// #include <TNL/String.h>

using namespace TNL;

TEST( FileNameTest, Constructor )
{
    FileName fname;

    EXPECT_EQ( fname.getFileName(), "00000." );
}

TEST( FileNameTest, Base )
{
    FileName fname;
    fname.setFileNameBase("name");

    EXPECT_EQ( fname.getFileName(), "name00000." );
}

TEST( FileNameTest, Extension )
{
    FileName fname;
    fname.setExtension("tnl");

    EXPECT_EQ( fname.getFileName(), "00000.tnl" );
}

TEST( FileNameTest, Index )
{
    FileName fname1;
    FileName fname2;
    fname1.setIndex(1);
    fname2.setIndex(50);

    EXPECT_EQ( fname1.getFileName(), "00001." );
    EXPECT_EQ( fname2.getFileName(), "00050." );
}

TEST( FileNameTest, DigitsCount )
{
    FileName fname;
    fname.setDigitsCount(4);

    EXPECT_EQ( fname.getFileName(), "0000." );
}

TEST( FileNameTest, AllTogether )
{
    FileName fname;
    fname.setFileNameBase("name");
    fname.setExtension("tnl");
    fname.setIndex(8);
    fname.setDigitsCount(3);

    EXPECT_EQ( fname.getFileName(), "name008.tnl" );
    EXPECT_EQ( getFileExtension(fname.getFileName()), "tnl" );
}
#endif

#include "main.h"
