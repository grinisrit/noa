#include <TNL/Arithmetics/Complex.h>

#include <complex>
#include <cmath>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Arithmetics;

#ifdef HAVE_GTEST

// test fixture for typed tests
template< typename Value >
class ComplexTest : public ::testing::Test
{
protected:
   using RealType = Value;
   using ComplexType = Complex< RealType >;

   static constexpr int testSize = 25;
};

// types for which unit tests of complex numbers are instantiated
using ComplexTypes = ::testing::Types< float, double >;

TYPED_TEST_SUITE( ComplexTest, ComplexTypes );

TYPED_TEST( ComplexTest, constructor )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType c;

   EXPECT_EQ( c.real(), ( RealType ) 0.0 );
   EXPECT_EQ( c.imag(), ( RealType ) 0.0 );
}

TYPED_TEST( ComplexTest, constructorWithRealValue )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType c{ ( RealType ) 1.0 };

   EXPECT_EQ( c.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c.imag(), ( RealType ) 0.0 );
}

TYPED_TEST( ComplexTest, constructorWithBothValues )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType c{ ( RealType ) 1.0, ( RealType ) 2.0 };

   EXPECT_EQ( c.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c.imag(), ( RealType ) 2.0 );
}

TYPED_TEST( ComplexTest, copyConstructors )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType u{ 1.0, 2.0 }, c{ u };

   EXPECT_EQ( c.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c.imag(), ( RealType ) 2.0 );

   Complex< int > iu( 1, 2 );
   ComplexType ci{ iu };

   EXPECT_EQ( ci.real(), ( RealType ) 1.0 );
   EXPECT_EQ( ci.imag(), ( RealType ) 2.0 );
}

TYPED_TEST( ComplexTest, constructorsFromSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   std::complex< RealType > std_c( 1.0, 2.0 );
   ComplexType c1( std_c );

   EXPECT_EQ( c1.real(), std_c.real() );
   EXPECT_EQ( c1.imag(), std_c.imag() );

   std::complex< int > std_int_c( 1, 2 );
   ComplexType c2( std_int_c );

   EXPECT_EQ( c2.real(), std_int_c.real() );
   EXPECT_EQ( c2.imag(), std_int_c.imag() );
}

TYPED_TEST( ComplexTest, assignmentWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType c1 = ( RealType ) 1.0;

   EXPECT_EQ( c1.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c1.imag(), ( RealType ) 0.0 );

   ComplexType c2 = 1;

   EXPECT_EQ( c2.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c2.imag(), ( RealType ) 0.0 );
}

TYPED_TEST( ComplexTest, assignmentWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   ComplexType c1( 1.0, 2.0 ), c2;
   c2 = c1;

   EXPECT_EQ( c2.real(), ( RealType ) 1.0 );
   EXPECT_EQ( c2.imag(), ( RealType ) 2.0 );

   Complex< int > ic1;
   ic1.real() = 1;
   ic1.imag() = 2;
   Complex ic2 = ic1;

   EXPECT_EQ( ic2.real(), ( RealType ) 1.0 );
   EXPECT_EQ( ic2.imag(), ( RealType ) 2.0 );
}

TYPED_TEST( ComplexTest, assignmentWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   std::complex< RealType > std_c( 1.0, 2.0 );
   ComplexType c1;
   c1 = std_c;

   EXPECT_EQ( c1.real(), std_c.real() );
   EXPECT_EQ( c1.imag(), std_c.imag() );

   std::complex< int > std_int_c( 1, 2 );
   ComplexType c2;
   c2 = std_int_c;

   EXPECT_EQ( c1.real(), std_int_c.real() );
   EXPECT_EQ( c1.imag(), std_int_c.imag() );
}

////
// Add assignment operator tests
TYPED_TEST( ComplexTest, addAssignmentOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci += cj.real();
         std_ci += std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, addAssignmentOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci += cj;
         std_ci += std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, addAssignmentOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci += std_cj;
         std_ci += std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

////
// Subtract assignment operator tests
TYPED_TEST( ComplexTest, subtractAssignmentOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci -= cj.real();
         std_ci -= std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, subtractAssignmentOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci -= cj;
         std_ci -= std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, subtractAssignmentOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci -= std_cj;
         std_ci -= std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

////
// Multiply assignment operator tests
TYPED_TEST( ComplexTest, multiplyAssignmentOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci *= cj.real();
         std_ci *= std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, multiplyAssignmentOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci *= cj;
         std_ci *= std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, multiplyAssignmentOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci *= std_cj;
         std_ci *= std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

////
// Divide assignment operator tests
TYPED_TEST( ComplexTest, divideAssignmentOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci /= cj.real();
         std_ci /= std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, divideAssignmentOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci /= cj;
         std_ci /= std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, divideAssignmentOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci /= std_cj;
         std_ci /= std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

////
// Comparison operator tests
TYPED_TEST( ComplexTest, comparisonOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      ComplexType ci( TNL::cos( phase_i ), 0.0 );
      ComplexType cj( TNL::cos( phase_i ), 1.0 );
      EXPECT_TRUE( ci == TNL::cos( phase_i ) );
      EXPECT_TRUE( cj != TNL::cos( phase_i ) );
   }
}

TYPED_TEST( ComplexTest, comparisonOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
      ComplexType cj( TNL::cos( phase_i ), TNL::sin( phase_i ) );
      ComplexType d1( TNL::cos( phase_i )+1.0, TNL::sin( phase_i ) );
      ComplexType d2( TNL::cos( phase_i ), TNL::sin( phase_i )+1.0 );
      ComplexType d3( TNL::cos( phase_i )+1.0, TNL::sin( phase_i )+1.0 );
      EXPECT_TRUE( ci == cj );
      EXPECT_TRUE( ci != d1 );
      EXPECT_TRUE( ci != d2 );
      EXPECT_TRUE( ci != d3 );
   }
}

TYPED_TEST( ComplexTest, comparisonOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
      std::complex< RealType > cj( TNL::cos( phase_i ), TNL::sin( phase_i ) );
      std::complex< RealType > d1( TNL::cos( phase_i )+1.0, TNL::sin( phase_i ) );
      std::complex< RealType > d2( TNL::cos( phase_i ), TNL::sin( phase_i )+1.0 );
      std::complex< RealType > d3( TNL::cos( phase_i )+1.0, TNL::sin( phase_i )+1.0 );
      EXPECT_TRUE( ci == cj );
      EXPECT_TRUE( ci != d1 );
      EXPECT_TRUE( ci != d2 );
      EXPECT_TRUE( ci != d3 );
   }
}

////
// Add operator tests
TYPED_TEST( ComplexTest, addOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci = ci + cj.real();
         std_ci = std_ci + std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, addOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci + cj;
         std_ci = std_ci + std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, addOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci + std_cj;
         std_ci = std_ci + std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

////
// Subtract operator tests
TYPED_TEST( ComplexTest, subtractOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci = ci - cj.real();
         std_ci =  std_ci - std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, subtractOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci - cj;
         std_ci = std_ci - std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

TYPED_TEST( ComplexTest, subtractOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci - std_cj;
         std_ci = std_ci - std_cj;
         EXPECT_EQ( ci.real(), std_ci.real() );
         EXPECT_EQ( ci.imag(), std_ci.imag() );
      }
}

////
// Multiply assignment operator tests
TYPED_TEST( ComplexTest, multiplyOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci = ci * cj.real();
         std_ci = std_ci * std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, multiplyOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci =  ci * cj;
         std_ci =  std_ci * std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, multiplyOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci * std_cj;
         std_ci = std_ci * std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

////
// Divide operator tests
TYPED_TEST( ComplexTest, divideOperatorWithReal )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), 0.0 );
         ci = ci / cj.real();
         std_ci = std_ci / std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, divideOperatorWithComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci / cj;
         std_ci = std_ci / std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, divideOperatorWithSTLComplex )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
      for( int j = 0; j <= size; j++ )
      {
         RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
         RealType phase_j = 2.0 * j * M_PI / ( RealType ) size;
         ComplexType ci( TNL::cos( phase_i ), TNL::sin( phase_i ) );
         ComplexType cj( TNL::cos( phase_j ), TNL::sin( phase_j ) );
         std::complex< RealType > std_ci( ci. real(), ci.imag() );
         std::complex< RealType > std_cj( cj. real(), cj.imag() );
         ci = ci / std_cj;
         std_ci = std_ci / std_cj;
         EXPECT_NEAR( ci.real(), std_ci.real(), 1.0e-6 );
         EXPECT_NEAR( ci.imag(), std_ci.imag(), 1.0e-6 );
      }
}

TYPED_TEST( ComplexTest, norm )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      RealType amp = 0.1 * ( RealType) i;
      ComplexType ci( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      std::complex< RealType > cj( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      EXPECT_EQ( norm( ci ), norm( cj ) );
   }
}

TYPED_TEST( ComplexTest, abs )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      RealType amp = 0.1 * ( RealType) i;
      ComplexType ci( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      std::complex< RealType > cj( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      EXPECT_NEAR( abs( ci ), abs( cj ), 1.0e-6 );
   }
}

TYPED_TEST( ComplexTest, arg )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      RealType amp = 0.1 * ( RealType) i;
      ComplexType ci( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      std::complex< RealType > cj( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      EXPECT_NEAR( arg( ci ), arg( cj ), 1.0e-6 );
   }
}

TYPED_TEST( ComplexTest, conj )
{
   using ComplexType = typename TestFixture::ComplexType;
   using RealType = typename ComplexType::ValueType;

   const int size = TestFixture::testSize;
   for( int i = 0; i <= size; i++ )
   {
      RealType phase_i = 2.0 * i * M_PI / ( RealType ) size;
      RealType amp = 0.1 * ( RealType) i;
      ComplexType ci( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      std::complex< RealType > cj( amp * TNL::cos( phase_i ), amp * TNL::sin( phase_i ) );
      EXPECT_EQ( conj( ci ).real(), conj( cj ).real() );
      EXPECT_EQ( conj( ci ).imag(), conj( cj ).imag() );
   }
}


#endif

#include "../main.h"
