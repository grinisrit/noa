#pragma once

#ifdef HAVE_GTEST

#if defined(DISTRIBUTED_VECTOR)
   #include <TNL/Containers/DistributedVector.h>
   #include <TNL/Containers/DistributedVectorView.h>
   #include <TNL/Containers/Partitioner.h>
   using namespace TNL::MPI;
#elif defined(STATIC_VECTOR)
   #include <TNL/Containers/StaticVector.h>
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
      #include <TNL/Containers/StaticVector.h>
   #endif
   #include <TNL/Containers/Vector.h>
   #include <TNL/Containers/VectorView.h>
#endif

#include "VectorHelperFunctions.h"
#include "../CustomScalar.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

namespace unary_tests {

// prime number to force non-uniform distribution in block-wise algorithms
constexpr int VECTOR_TEST_SIZE = 97;

// test fixture for typed tests
template< typename T >
class VectorUnaryOperationsTest : public ::testing::Test
{
protected:
   using VectorOrView = T;
#ifdef STATIC_VECTOR
   template< typename Real >
   using Vector = StaticVector< VectorOrView::getSize(), Real >;
#else
   using NonConstReal = std::remove_const_t< typename VectorOrView::RealType >;
   #ifdef DISTRIBUTED_VECTOR
      using VectorType = DistributedVector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
      template< typename Real >
      using Vector = DistributedVector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;

      const MPI_Comm communicator = MPI_COMM_WORLD;

      const int rank = GetRank(communicator);
      const int nproc = GetSize(communicator);

      // some arbitrary even value (but must be 0 if not distributed)
      const int ghosts = (nproc > 1) ? 4 : 0;
   #else
      using VectorType = Containers::Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
      template< typename Real >
      using Vector = Containers::Vector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   #endif
#endif
};

// types for which VectorUnaryOperationsTest is instantiated
#if defined(DISTRIBUTED_VECTOR)
   using VectorTypes = ::testing::Types<
   #ifndef __CUDACC__
      DistributedVector<           double, Devices::Host, int >,
      DistributedVectorView<       double, Devices::Host, int >,
      DistributedVectorView< const double, Devices::Host, int >,
      DistributedVector< CustomScalar< double >, Devices::Host, int >
   #else
      DistributedVector<           double, Devices::Cuda, int >,
      DistributedVectorView<       double, Devices::Cuda, int >,
      DistributedVectorView< const double, Devices::Cuda, int >,
      DistributedVector< CustomScalar< double >, Devices::Cuda, int >
   #endif
   >;
#elif defined(STATIC_VECTOR)
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorTypes = ::testing::Types<
         StaticVector< 1, StaticVector< 3, double > >,
         StaticVector< 2, StaticVector< 3, double > >,
         StaticVector< 3, StaticVector< 3, double > >,
         StaticVector< 4, StaticVector< 3, double > >,
         StaticVector< 5, StaticVector< 3, CustomScalar< double > > >
      >;
   #else
      using VectorTypes = ::testing::Types<
         StaticVector< 1, int >,
         StaticVector< 1, double >,
         StaticVector< 2, int >,
         StaticVector< 2, double >,
         StaticVector< 3, int >,
         StaticVector< 3, double >,
         StaticVector< 4, int >,
         StaticVector< 4, double >,
         StaticVector< 5, CustomScalar< int > >,
         StaticVector< 5, CustomScalar< double > >
      >;
   #endif
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorTypes = ::testing::Types<
      #ifndef __CUDACC__
         Vector<     StaticVector< 3, double >, Devices::Host >,
         VectorView< StaticVector< 3, double >, Devices::Host >,
         VectorView< StaticVector< 3, CustomScalar< double > >, Devices::Host >
      #else
         Vector<     StaticVector< 3, double >, Devices::Cuda >,
         VectorView< StaticVector< 3, double >, Devices::Cuda >,
         VectorView< StaticVector< 3, CustomScalar< double > >, Devices::Cuda >
      #endif
      >;
   #else
      using VectorTypes = ::testing::Types<
      #ifndef __CUDACC__
         Vector<     int,       Devices::Host >,
         VectorView< int,       Devices::Host >,
         VectorView< const int, Devices::Host >,
         Vector<     double,    Devices::Host >,
         VectorView< double,    Devices::Host >,
         Vector<     CustomScalar< int >, Devices::Host >,
         VectorView< CustomScalar< int >, Devices::Host >
      #endif
      #ifdef __CUDACC__
         Vector<     int,       Devices::Cuda >,
         VectorView< int,       Devices::Cuda >,
         VectorView< const int, Devices::Cuda >,
         Vector<     double,    Devices::Cuda >,
         VectorView< double,    Devices::Cuda >,
         Vector<     CustomScalar< int >, Devices::Cuda >,
         VectorView< CustomScalar< int >, Devices::Cuda >
      #endif
      >;
   #endif
#endif

TYPED_TEST_SUITE( VectorUnaryOperationsTest, VectorTypes );


#define EXPECTED_VECTOR( TestFixture, function ) \
   using ExpectedVector = typename TestFixture::template Vector< Expressions::RemoveET< decltype(function(typename VectorOrView::RealType{})) > >;

#ifdef STATIC_VECTOR
   #define SETUP_UNARY_VECTOR_TEST( _ ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
                                                               \
      VectorOrView V1, V2;                                     \
                                                               \
      V1 = 1;                                                  \
      V2 = 2;                                                  \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( _, begin, end, function ) \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorOrView::RealType;        \
      EXPECTED_VECTOR( TestFixture, function );                \
      constexpr int _size = VectorOrView::getSize();           \
                                                               \
      VectorOrView V1;                                         \
      ExpectedVector expected;                                 \
                                                               \
      const double h = (double) (end - begin) / _size;         \
      for( int i = 0; i < _size; i++ )                         \
      {                                                        \
         const RealType x = begin + RealType( i * h );         \
         V1[ i ] = x;                                          \
         expected[ i ] = function(x);                          \
      }                                                        \
      (void) 0  // dummy statement here enforces ';' after the macro use

#elif defined(DISTRIBUTED_VECTOR)
   #define SETUP_UNARY_VECTOR_TEST( size ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType >::splitRange( size, this->communicator ); \
      using Synchronizer = typename Partitioner< typename VectorOrView::IndexType >::template ArraySynchronizer< typename VectorOrView::DeviceType >; \
                                                               \
      VectorType _V1, _V2;                                     \
      _V1.setDistribution( localRange, this->ghosts, size, this->communicator ); \
      _V2.setDistribution( localRange, this->ghosts, size, this->communicator ); \
                                                               \
      auto _synchronizer = std::make_shared<Synchronizer>( localRange, this->ghosts / 2, this->communicator ); \
      _V1.setSynchronizer( _synchronizer );                    \
      _V2.setSynchronizer( _synchronizer );                    \
                                                               \
      _V1 = 1;                                                 \
      _V2 = 2;                                                 \
                                                               \
      VectorOrView V1( _V1 ), V2( _V2 );                       \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorType::RealType;          \
      EXPECTED_VECTOR( TestFixture, function );                \
      using HostVector = typename VectorType::template Self< RealType, Devices::Host >; \
      using HostExpectedVector = typename ExpectedVector::template Self< typename ExpectedVector::RealType, Devices::Host >; \
      using LocalRangeType = typename VectorOrView::LocalRangeType; \
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType >::splitRange( size, this->communicator ); \
      using Synchronizer = typename Partitioner< typename VectorOrView::IndexType >::template ArraySynchronizer< typename VectorOrView::DeviceType >; \
                                                               \
      HostVector _V1h;                                         \
      HostExpectedVector expected_h;                           \
      _V1h.setDistribution( localRange, this->ghosts, size, this->communicator ); \
      expected_h.setDistribution( localRange, this->ghosts, size, this->communicator ); \
                                                               \
      const double h = (double) (end - begin) / size;          \
      for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ ) \
      {                                                        \
         const RealType x = begin + RealType( i * h );         \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
      for( int i = localRange.getSize(); i < _V1h.getLocalView().getSize(); i++ ) \
         _V1h.getLocalView()[ i ] = expected_h.getLocalView()[ i ] = 0;           \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      ExpectedVector expected; expected = expected_h;          \
                                                               \
      auto _synchronizer = std::make_shared<Synchronizer>( localRange, this->ghosts / 2, this->communicator ); \
      _V1.setSynchronizer( _synchronizer );                    \
      expected.setSynchronizer( _synchronizer );               \
      expected.startSynchronization();                         \
      (void) 0  // dummy statement here enforces ';' after the macro use

#else
   #define SETUP_UNARY_VECTOR_TEST( size ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using ValueType = typename VectorType::ValueType;        \
                                                               \
      VectorType _V1( size ), _V2( size );                     \
                                                               \
      _V1 = ValueType( 1 );                                    \
      _V2 = ValueType( 2 );                                    \
                                                               \
      VectorOrView V1( _V1 ), V2( _V2 );                       \
      (void) 0  // dummy statement here enforces ';' after the macro use

   #define SETUP_UNARY_VECTOR_TEST_FUNCTION( size, begin, end, function ) \
      using VectorType = typename TestFixture::VectorType;     \
      using VectorOrView = typename TestFixture::VectorOrView; \
      using RealType = typename VectorType::RealType;          \
      EXPECTED_VECTOR( TestFixture, function );                \
      using HostVector = typename VectorType::template Self< RealType, Devices::Host >; \
      using HostExpectedVector = typename ExpectedVector::template Self< typename ExpectedVector::RealType, Devices::Host >; \
                                                               \
      HostVector _V1h( size );                                 \
      HostExpectedVector expected_h( size );                   \
                                                               \
      const double h = (double) (end - begin) / size;          \
      for( int i = 0; i < size; i++ )                          \
      {                                                        \
         const RealType x = begin + RealType( i * h );         \
         _V1h[ i ] = x;                                        \
         expected_h[ i ] = function(x);                        \
      }                                                        \
                                                               \
      VectorType _V1; _V1 = _V1h;                              \
      VectorOrView V1( _V1 );                                  \
      ExpectedVector expected; expected = expected_h;          \
      (void) 0  // dummy statement here enforces ';' after the macro use

#endif

// This is because exact comparison does not work due to rounding errors:
// - the "expected" vector is computed sequentially on CPU
// - the host compiler might decide to use a vectorized version of the
//   math function, which may have slightly different precision
// - GPU may have different precision than CPU, so exact comparison with
//   the result from host is not possible
template< typename Left, typename Right >
void expect_vectors_near( const Left& _v1, const Right& _v2 )
{
   ASSERT_EQ( _v1.getSize(), _v2.getSize() );
#ifdef STATIC_VECTOR
   for( int i = 0; i < _v1.getSize(); i++ )
      expect_near( _v1[i], _v2[i], 1e-6 );
#else
   using LeftNonConstReal = Expressions::RemoveET< std::remove_const_t< typename Left::RealType > >;
   using RightNonConstReal = Expressions::RemoveET< std::remove_const_t< typename Right::RealType > >;
#ifdef DISTRIBUTED_VECTOR
   using LeftVector = DistributedVector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = DistributedVector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
#else
   using LeftVector = Vector< LeftNonConstReal, typename Left::DeviceType, typename Left::IndexType >;
   using RightVector = Vector< RightNonConstReal, typename Right::DeviceType, typename Right::IndexType >;
#endif
   using LeftHostVector = typename LeftVector::template Self< LeftNonConstReal, Devices::Sequential >;
   using RightHostVector = typename RightVector::template Self< RightNonConstReal, Devices::Sequential >;

   // first evaluate expressions
   LeftVector v1; v1 = _v1;
   RightVector v2; v2 = _v2;
   // then copy to host
   LeftHostVector v1_h; v1_h = v1;
   RightHostVector v2_h; v2_h = v2;
#ifdef DISTRIBUTED_VECTOR
   const auto localRange = v1.getLocalRange();
   for( int i = localRange.getBegin(); i < localRange.getEnd(); i++ )
#else
   for( int i = 0; i < v1.getSize(); i++ )
#endif
      expect_near( v1_h[i], v2_h[i], 1e-6 );
#endif
}

TYPED_TEST( VectorUnaryOperationsTest, plus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( +V1, 1 );
   // unary expression
   EXPECT_EQ( +(+V2), 2 );
   // binary expression
   EXPECT_EQ( +(V1 + V1), 2 );
}

TYPED_TEST( VectorUnaryOperationsTest, minus )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( -V1, -1 );
   // unary expression
   EXPECT_EQ( -(-V2), 2 );
   // binary expression
   EXPECT_EQ( -(V1 + V1), -2 );
}

TYPED_TEST( VectorUnaryOperationsTest, logicalNot )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( !V1, 0 );
   // unary expression
   EXPECT_EQ( !(!V2), 1 );
   // binary expression
   EXPECT_EQ( !(V1 + V1), 0 );
}

TYPED_TEST( VectorUnaryOperationsTest, bitNot )
{
   // binary negation is defined only for integral types
   using ValueType = typename TestFixture::VectorOrView::ValueType;
   if constexpr( std::is_integral_v< ValueType > ) {
      SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

      // vector or view
      EXPECT_EQ( ~V1, ~static_cast< ValueType >( 1 ) );
      // unary expression
      EXPECT_EQ( ~(~V2), 2 );
      // binary expression
      EXPECT_EQ( ~(V1 + V1), ~static_cast< ValueType >( 2 ) );
   }
}

TYPED_TEST( VectorUnaryOperationsTest, abs )
{
   SETUP_UNARY_VECTOR_TEST( VECTOR_TEST_SIZE );

   // vector or view
   EXPECT_EQ( abs(V1), V1 );
   // unary expression
   EXPECT_EQ( abs(-V1), V1 );
   // binary expression
   EXPECT_EQ( abs(-V1-V1), V2 );
}

TYPED_TEST( VectorUnaryOperationsTest, sin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sin );

   // vector or view
   expect_vectors_near( sin(V1), expected );
   // binary expression
   expect_vectors_near( sin(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sin(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asin )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::asin );

   // vector or view
   expect_vectors_near( asin(V1), expected );
   // binary expression
   expect_vectors_near( asin(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( asin(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cos );

   // vector or view
   expect_vectors_near( cos(V1), expected );
   // binary expression
   expect_vectors_near( cos(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cos(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acos )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.0, 1.0, TNL::acos );

   // vector or view
   expect_vectors_near( acos(V1), expected );
   // binary expression
   expect_vectors_near( acos(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( acos(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -1.5, 1.5, TNL::tan );

   // vector or view
   expect_vectors_near( tan(V1), expected );
   // binary expression
   expect_vectors_near( tan(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( tan(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atan )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::atan );

   // vector or view
   expect_vectors_near( atan(V1), expected );
   // binary expression
   expect_vectors_near( atan(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( atan(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::sinh );

   // vector or view
   expect_vectors_near( sinh(V1), expected );
   // binary expression
   expect_vectors_near( sinh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sinh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, asinh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::asinh );

   // vector or view
   expect_vectors_near( asinh(V1), expected );
   // binary expression
   expect_vectors_near( asinh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( asinh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::cosh );

   // vector or view
   expect_vectors_near( cosh(V1), expected );
   // binary expression
   expect_vectors_near( cosh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cosh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, acosh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::acosh );

   // vector or view
   expect_vectors_near( acosh(V1), expected );
   // binary expression
   expect_vectors_near( acosh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( acosh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, tanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::tanh );

   // vector or view
   expect_vectors_near( tanh(V1), expected );
   // binary expression
   expect_vectors_near( tanh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( tanh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, atanh )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -0.99, 0.99, TNL::atanh );

   // vector or view
   expect_vectors_near( atanh(V1), expected );
   // binary expression
   expect_vectors_near( atanh(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( atanh(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, pow )
{
   // FIXME: for integer exponent, the test fails with CUDA
//   auto pow3 = [](auto i) { return TNL::pow(i, 3); };
   auto pow3 = [](auto i) { return TNL::pow(i, 3.0); };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, pow3 );

   // vector or view
   expect_vectors_near( pow(V1, 3.0), expected );
   // binary expression
   expect_vectors_near( pow(2 * V1 - V1, 3.0), expected );
   // unary expression
   expect_vectors_near( pow(-(-V1), 3.0), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, exp )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -10, 10, TNL::exp );

   // vector or view
   expect_vectors_near( exp(V1), expected );
   // binary expression
   expect_vectors_near( exp(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( exp(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sqrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 0, VECTOR_TEST_SIZE, TNL::sqrt );

   // vector or view
   expect_vectors_near( sqrt(V1), expected );
   // binary expression
   expect_vectors_near( sqrt(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sqrt(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, cbrt )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::cbrt );

   // vector or view
   expect_vectors_near( cbrt(V1), expected );
   // binary expression
   expect_vectors_near( cbrt(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( cbrt(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log );

   // vector or view
   expect_vectors_near( log(V1), expected );
   // binary expression
   expect_vectors_near( log(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log10 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log10 );

   // vector or view
   expect_vectors_near( log10(V1), expected );
   // binary expression
   expect_vectors_near( log10(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log10(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, log2 )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, TNL::log2 );

   // vector or view
   expect_vectors_near( log2(V1), expected );
   // binary expression
   expect_vectors_near( log2(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( log2(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, floor )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::floor );

   // vector or view
   expect_vectors_near( floor(V1), expected );
   // binary expression
   expect_vectors_near( floor(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( floor(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, ceil )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -3.0, 3.0, TNL::ceil );

   // vector or view
   expect_vectors_near( ceil(V1), expected );
   // binary expression
   expect_vectors_near( ceil(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( ceil(-(-V1)), expected );
}

TYPED_TEST( VectorUnaryOperationsTest, sign )
{
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, -VECTOR_TEST_SIZE, VECTOR_TEST_SIZE, TNL::sign );

   // vector or view
   expect_vectors_near( sign(V1), expected );
   // binary expression
   expect_vectors_near( sign(2 * V1 - V1), expected );
   // unary expression
   expect_vectors_near( sign(-(-V1)), expected );
}

// This test is not suitable for vector-of-static-vectors where the RealType cannot be cast to bool.
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorUnaryOperationsTest, cast )
{
   auto identity = [](auto i) { return i; };
   SETUP_UNARY_VECTOR_TEST_FUNCTION( VECTOR_TEST_SIZE, 1, VECTOR_TEST_SIZE, identity );

   // vector or vector view
   auto expression1 = cast<bool>(V1);
   static_assert( std::is_same< typename decltype(expression1)::RealType, bool >::value,
                  "BUG: the cast function does not work for vector or vector view." );
   EXPECT_EQ( expression1, true );

   // binary expression
   auto expression2 = cast<bool>(V1 + V1);
   static_assert( std::is_same< typename decltype(expression2)::RealType, bool >::value,
                  "BUG: the cast function does not work for binary expression." );
   // FIXME: expression2 cannot be reused, because expression templates for StaticVector and DistributedVector contain references and the test would crash in Release
//   EXPECT_EQ( expression2, true );
   EXPECT_EQ( cast<bool>(V1 + V1), true );

   // unary expression
   auto expression3 = cast<bool>(-V1);
   static_assert( std::is_same< typename decltype(expression3)::RealType, bool >::value,
                  "BUG: the cast function does not work for unary expression." );
   // FIXME: expression3 cannot be reused, because expression templates for StaticVector and DistributedVector contain references and the test would crash in Release
//   EXPECT_EQ( expression3, true );
   EXPECT_EQ( cast<bool>(-V1), true );
}
#endif

} // namespace unary_tests

#endif // HAVE_GTEST
