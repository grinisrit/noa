/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Daniel Simon, dansimon93@gmail.com
 */

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#ifdef HAVE_GMP
#include <gmp.h>
#endif

#include <TNL/Arithmetics/MultiPrecision.h>

/*NUMBERS*/
#define PREC 128
#define num_1 0.0102030405
#define num_2 1.1234567891

/*
 INFO:
 This test compares values from the GMP Library with values from our wrapped GMP Library in MultiPrecision.
 MP_res -> result from MultiPrecision
 GMP_res -> result from GMP Library
 */

using namespace TNL;
using namespace TNL::Arithmetics;

#ifdef HAVE_GTEST
TEST (MultiPrecisionTest, number_assignment)
{
    /* GMPLIB */
    mpf_t mpf1;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_1);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_1);

    EXPECT_EQ (mp1 , mpf1);
}


TEST (MultiPrecisionTest, number_negation)
{
    /* GMPLIB */
    mpf_t mpf1, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_1);
    mpf_init (GMP_res);
    mpf_neg (GMP_res , mpf1);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_1);
    MultiPrecision MP_res (-mp1);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_plus_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_add (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 += mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_minus_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_sub (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 -= mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_mul_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_mul (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 *= mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_div_equals)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_div (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 /= mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_plus)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_add (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 + mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_minus)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_sub (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 - mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_div)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_div (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 / mp2);

    EXPECT_EQ (MP_res , GMP_res);
}


TEST (MultiPrecisionTest, op_mul)
{
    /* GMPLIB */
    mpf_t mpf1, mpf2, GMP_res;
    mpf_set_default_prec (PREC);
    mpf_init_set_d (mpf1 , num_2);
    mpf_init_set_d (mpf2 , num_1);
    mpf_init (GMP_res);
    mpf_mul (GMP_res , mpf1 , mpf2);

    /* MultiPrecision */
    MultiPrecision::setPrecision(PREC);
    MultiPrecision mp1 (num_2);
    MultiPrecision mp2 (num_1);
    MultiPrecision MP_res (mp1 * mp2);

    EXPECT_EQ (MP_res , GMP_res);
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
