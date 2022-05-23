/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Daniel Simon, dansimon93@gmail.com
 */

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/Arithmetics/MultiPrecision.h>
#include <TNL/Arithmetics/Quad.h>

/*NUMBERS*/
#define num_1 2.1230405067890102030405060708096352410708
#define num_2 1.2080706050401236549873571590082467951301

/*
 INFO:
 This test compares values from MultiPrecision with Quad (Quadruple precision)
 MP_res -> result from MultiPrecision
 QD_res -> result from Quad
 */

using namespace TNL;
using namespace TNL::Arithmetics;

#if ( defined HAVE_GTEST ) && ( defined HAVE_GMP )
TEST (QuadTest, number_assignment)
{
    /* Quad */
    Quad<double> qd1 (num_1);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);

    EXPECT_EQ (mp1 , qd1);
}


TEST (QuadTest, op_plus_equals)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 += qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 += mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_minus_equals)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 -= qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 -= mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_mul_equals)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 *= qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 *= mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_div_equals)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 /= qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 /= mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_plus)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 + qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 + mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_minus)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 - qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 - mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_mul)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 * qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 * mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, op_div)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    Quad<double> QD_res (qd1 / qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    MultiPrecision MP_res (mp1 / mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_equal)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 == qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 == mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_not_equal)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 != qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 != mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_less)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 < qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 < mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_greater)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 > qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 > mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_greater_equal)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 >= qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 >= mp2);

    EXPECT_EQ (MP_res , QD_res);
}


TEST (QuadTest, cmp_less_equal)
{
    /* Quad */
    Quad<double> qd1 (num_1);
    Quad<double> qd2 (num_2);
    bool QD_res (qd1 <= qd2);

    /* MultiPrecision */
    MultiPrecision mp1 (num_1);
    MultiPrecision mp2 (num_2);
    bool MP_res (mp1 <= mp2);

    EXPECT_EQ (MP_res , QD_res);
}
#endif


#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
