#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#include <TNL/Meshes/DistributedMeshes/Directions.h>
#include <TNL/Containers/StaticVector.h>

using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Containers;
using namespace TNL;

TEST(Direction1D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),ZzYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),ZzYzXp) << "Failed";
}

TEST(Direction2D, Edge)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),ZzYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),ZzYzXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-2)),ZzYmXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(2)),ZzYpXz) << "Failed";

    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-1,0)),ZzYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(1,0)),ZzYzXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(0,-2)),ZzYmXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(0,2)),ZzYpXz) << "Failed";
}

TEST(Direction2D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2,-1)),ZzYmXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2, 1)),ZzYmXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(2,-1)),ZzYpXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(2,1)),ZzYpXp) << "Failed";
}

TEST(Direction3D, Faces)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-1)),ZzYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(1)),ZzYzXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-2)),ZzYmXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(2)),ZzYpXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(-3)),ZmYzXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<1,int>(3)),ZpYzXz) << "Failed";
}

TEST(Direction3D, Edges)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2,-1)),ZzYmXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-2, 1)),ZzYmXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 2,-1)),ZzYpXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 2, 1)),ZzYpXp) << "Failed";
    
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3,-1)),ZmYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3, 1)),ZmYzXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3,-2)),ZmYmXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>(-3, 2)),ZmYpXz) << "Failed";

    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3,-1)),ZpYzXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3, 1)),ZpYzXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3,-2)),ZpYmXz) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<2,int>( 3, 2)),ZpYpXz) << "Failed";
}

TEST(Direction3D, Conners)
{
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3,-2,-1)),ZmYmXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3,-2, 1)),ZmYmXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3, 2,-1)),ZmYpXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(-3, 2, 1)),ZmYpXp) << "Failed";
    
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3,-2,-1)),ZpYmXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3,-2, 1)),ZpYmXp) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3, 2,-1)),ZpYpXm) << "Failed";
    EXPECT_EQ(Directions::getDirection(StaticVector<3,int>(3, 2, 1)),ZpYpXp) << "Failed";
}

TEST(XYZ, 2D )
{
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYzXm), (StaticVector<2,int>(-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYzXp), (StaticVector<2,int>(1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYmXz), (StaticVector<2,int>(0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYpXz), (StaticVector<2,int>(0,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<2>(ZzYmXm), (StaticVector<2,int>(-1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYmXp), (StaticVector<2,int>(1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYpXm), (StaticVector<2,int>(-1,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<2>(ZzYpXp), (StaticVector<2,int>(1,1)) ) << "Failed";

}

TEST(XYZ, 3D )
{
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYzXm), (StaticVector<3,int>(-1,0,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYzXp), (StaticVector<3,int>(1,0,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYmXz), (StaticVector<3,int>(0,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYpXz), (StaticVector<3,int>(0,1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZmYzXz), (StaticVector<3,int>(0,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZpYzXz), (StaticVector<3,int>(0,0,1)) ) << "Failed";    

    EXPECT_EQ( Directions::template getXYZ<3>(ZzYmXm), (StaticVector<3,int>(-1,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYmXp), (StaticVector<3,int>(1,-1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYpXm), (StaticVector<3,int>(-1,1,0)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZzYpXp), (StaticVector<3,int>(1,1,0)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(ZmYzXm), (StaticVector<3,int>(-1,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZmYzXp), (StaticVector<3,int>(1,0,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZpYzXm), (StaticVector<3,int>(-1,0,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZpYzXp), (StaticVector<3,int>(1,0,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(ZmYmXz), (StaticVector<3,int>(0,-1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZmYpXz), (StaticVector<3,int>(0,1,-1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZpYmXz), (StaticVector<3,int>(0,-1,1)) ) << "Failed";
    EXPECT_EQ( Directions::template getXYZ<3>(ZpYpXz), (StaticVector<3,int>(0,1,1)) ) << "Failed";

    EXPECT_EQ( Directions::template getXYZ<3>(ZpYpXm), (StaticVector<3,int>(-1,1,1)) ) << "Failed";

}

#endif

#include "../../main.h"
