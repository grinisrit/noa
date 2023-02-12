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

namespace binary_tests {

// prime number to force non-uniform distribution in block-wise algorithms
constexpr int VECTOR_TEST_SIZE = 97;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_REDUCTION_SIZE = 4999;

template< typename T1, typename T2 >
struct Pair
{
   using Left = T1;
   using Right = T2;
};

// test fixture for typed tests
template< typename Pair >
class VectorBinaryOperationsTest : public ::testing::Test
{
protected:
   using Left = typename Pair::Left;
   using Right = typename Pair::Right;
   using LeftReal = std::remove_const_t< typename Left::RealType >;
   using RightReal = std::remove_const_t< typename Right::RealType >;
#ifndef STATIC_VECTOR
   #ifdef DISTRIBUTED_VECTOR
      using LeftVector = DistributedVector< LeftReal, typename Left::DeviceType, typename Left::IndexType >;
      using RightVector = DistributedVector< RightReal, typename Right::DeviceType, typename Right::IndexType >;

      const MPI_Comm communicator = MPI_COMM_WORLD;

      const int rank = GetRank(communicator);
      const int nproc = GetSize(communicator);

      // some arbitrary value (but must be 0 if not distributed)
      const int ghosts = (nproc > 1) ? 4 : 0;
   #else
      using LeftVector = Vector< LeftReal, typename Left::DeviceType, typename Left::IndexType >;
      using RightVector = Vector< RightReal, typename Right::DeviceType, typename Right::IndexType >;
   #endif
#endif

   Left L1, L2;
   Right R1, R2;

#ifndef STATIC_VECTOR
   LeftVector _L1, _L2;
   RightVector _R1, _R2;
#endif

   void reset( int size )
   {
#ifdef STATIC_VECTOR
      L1 = 1;
      L2 = 2;
      R1 = 1;
      R2 = 2;
#else
   #ifdef DISTRIBUTED_VECTOR
      using LocalRangeType = typename LeftVector::LocalRangeType;
      using Synchronizer = typename Partitioner< typename Left::IndexType >::template ArraySynchronizer< typename Left::DeviceType >;
      const LocalRangeType localRange = Partitioner< typename Left::IndexType >::splitRange( size, communicator );

      _L1.setDistribution( localRange, ghosts, size, communicator );
      _L2.setDistribution( localRange, ghosts, size, communicator );
      _R1.setDistribution( localRange, ghosts, size, communicator );
      _R2.setDistribution( localRange, ghosts, size, communicator );

      auto synchronizer = std::make_shared<Synchronizer>( localRange, ghosts / 2, communicator );
      _L1.setSynchronizer( synchronizer );
      _L2.setSynchronizer( synchronizer );
      _R1.setSynchronizer( synchronizer );
      _R2.setSynchronizer( synchronizer );
   #else
      _L1.setSize( size );
      _L2.setSize( size );
      _R1.setSize( size );
      _R2.setSize( size );
   #endif
      _L1 = 1;
      _L2 = 2;
      _R1 = 1;
      _R2 = 2;
      bindOrAssign( L1, _L1 );
      bindOrAssign( L2, _L2 );
      bindOrAssign( R1, _R1 );
      bindOrAssign( R2, _R2 );
#endif
   }

   VectorBinaryOperationsTest()
   {
      reset( VECTOR_TEST_SIZE );
   }
};

#ifdef __CUDACC__
   // ignore useless nvcc warning: https://stackoverflow.com/a/49997636
   // https://developer.nvidia.com/blog/reducing-application-build-times-using-cuda-c-compilation-aids/
   #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
      #pragma nv_diagnostic push
      #pragma nv_diag_suppress declared_but_not_referenced
   #else
      #pragma push
      #pragma diag_suppress = declared_but_not_referenced
   #endif
#endif

#define MAYBE_UNUSED(expr) (void)(expr)

#define SETUP_BINARY_TEST_ALIASES \
   using Left = typename TestFixture::Left;                 \
   using Right = typename TestFixture::Right;               \
   using LeftReal [[maybe_unused]] = typename TestFixture::LeftReal; \
   using RightReal [[maybe_unused]] = typename TestFixture::RightReal; \
   Left& L1 = this->L1;                                     \
   Left& L2 = this->L2;                                     \
   Right& R1 = this->R1;                                    \
   Right& R2 = this->R2;                                    \
   MAYBE_UNUSED(L1);                                        \
   MAYBE_UNUSED(L2);                                        \
   MAYBE_UNUSED(R1);                                        \
   MAYBE_UNUSED(R2);                                        \
   (void) 0  // dummy statement here enforces ';' after the macro use

// types for which VectorBinaryOperationsTest is instantiated
#if defined(DISTRIBUTED_VECTOR)
   using VectorPairs = ::testing::Types<
   #ifndef __CUDACC__
      Pair< DistributedVector<     int,   Devices::Host, int >,
            DistributedVector<     short, Devices::Host, int > >,
      Pair< DistributedVector<     int,   Devices::Host, int >,
            DistributedVectorView< short, Devices::Host, int > >,
      Pair< DistributedVectorView< int,   Devices::Host, int >,
            DistributedVector<     short, Devices::Host, int > >,
      Pair< DistributedVectorView< CustomScalar< int >,   Devices::Host, int >,
            DistributedVectorView< CustomScalar< short >, Devices::Host, int > >
   #else
      Pair< DistributedVector<     int,   Devices::Cuda, int >,
            DistributedVector<     short, Devices::Cuda, int > >,
      Pair< DistributedVector<     int,   Devices::Cuda, int >,
            DistributedVectorView< short, Devices::Cuda, int > >,
      Pair< DistributedVectorView< int,   Devices::Cuda, int >,
            DistributedVector<     short, Devices::Cuda, int > >,
      Pair< DistributedVectorView< CustomScalar< int >,   Devices::Cuda, int >,
            DistributedVectorView< CustomScalar< short >, Devices::Cuda, int > >
   #endif
   >;
#elif defined(STATIC_VECTOR)
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorPairs = ::testing::Types<
         Pair< StaticVector< 1, StaticVector< 3, int > >,  StaticVector< 1, StaticVector< 3, short > > >,
         Pair< StaticVector< 2, StaticVector< 3, int > >,  StaticVector< 2, StaticVector< 3, short > > >,
         Pair< StaticVector< 3, StaticVector< 3, int > >,  StaticVector< 3, StaticVector< 3, short > > >,
         Pair< StaticVector< 4, StaticVector< 3, int > >,  StaticVector< 4, StaticVector< 3, short > > >,
         Pair< StaticVector< 5, StaticVector< 3, int > >,  StaticVector< 5, StaticVector< 3, short > > >,
         Pair< StaticVector< 5, StaticVector< 3, CustomScalar< int > > >,  StaticVector< 5, StaticVector< 3, CustomScalar< short > > > >
      >;
   #else
      using VectorPairs = ::testing::Types<
         Pair< StaticVector< 1, int    >,  StaticVector< 1, short  > >,
         Pair< StaticVector< 1, double >,  StaticVector< 1, double > >,
         Pair< StaticVector< 2, int    >,  StaticVector< 2, short  > >,
         Pair< StaticVector< 2, double >,  StaticVector< 2, double > >,
         Pair< StaticVector< 3, int    >,  StaticVector< 3, short  > >,
         Pair< StaticVector< 3, double >,  StaticVector< 3, double > >,
         Pair< StaticVector< 4, int    >,  StaticVector< 4, short  > >,
         Pair< StaticVector< 4, double >,  StaticVector< 4, double > >,
         Pair< StaticVector< 5, int    >,  StaticVector< 5, CustomScalar< short > > >,
         Pair< StaticVector< 5, double >,  StaticVector< 5, CustomScalar< double > > >
      >;
   #endif
#else
   #ifdef VECTOR_OF_STATIC_VECTORS
      using VectorPairs = ::testing::Types<
      #ifndef __CUDACC__
         Pair< Vector<     StaticVector< 3, int >, Devices::Host >, Vector<     StaticVector< 3, short >, Devices::Host > >,
         Pair< VectorView< StaticVector< 3, int >, Devices::Host >, Vector<     StaticVector< 3, short >, Devices::Host > >,
         Pair< Vector<     StaticVector< 3, int >, Devices::Host >, VectorView< StaticVector< 3, short >, Devices::Host > >,
         Pair< VectorView< StaticVector< 3, int >, Devices::Host >, VectorView< StaticVector< 3, short >, Devices::Host > >
      #else
         Pair< Vector<     StaticVector< 3, int >, Devices::Cuda >, Vector<     StaticVector< 3, short >, Devices::Cuda > >,
         Pair< VectorView< StaticVector< 3, int >, Devices::Cuda >, Vector<     StaticVector< 3, short >, Devices::Cuda > >,
         Pair< Vector<     StaticVector< 3, int >, Devices::Cuda >, VectorView< StaticVector< 3, short >, Devices::Cuda > >,
         Pair< VectorView< StaticVector< 3, int >, Devices::Cuda >, VectorView< StaticVector< 3, short >, Devices::Cuda > >
      #endif
      >;
   #else
      using VectorPairs = ::testing::Types<
      #ifndef __CUDACC__
         Pair< Vector<     int,                 Devices::Host >, Vector<     int,                          Devices::Host > >,
         Pair< VectorView< int,                 Devices::Host >, Vector<     int,                          Devices::Host > >,
         Pair< VectorView< const int,           Devices::Host >, Vector<     int,                          Devices::Host > >,
         Pair< Vector<     CustomScalar< int >, Devices::Host >, VectorView< CustomScalar< double >,       Devices::Host > >,
         Pair< Vector<     CustomScalar< int >, Devices::Host >, VectorView< const CustomScalar< double >, Devices::Host > >,
         Pair< VectorView< CustomScalar< int >, Devices::Host >, VectorView< CustomScalar< double >,       Devices::Host > >,
         Pair< VectorView< const int,           Devices::Host >, VectorView< int,                          Devices::Host > >,
         Pair< VectorView< const int,           Devices::Host >, VectorView< const int,                    Devices::Host > >,
         Pair< VectorView< int,                 Devices::Host >, VectorView< const int,                    Devices::Host > >
      #else
         Pair< Vector<     int,                 Devices::Cuda >, Vector<     int,                          Devices::Cuda > >,
         Pair< VectorView< int,                 Devices::Cuda >, Vector<     int,                          Devices::Cuda > >,
         Pair< VectorView< const int,           Devices::Cuda >, Vector<     int,                          Devices::Cuda > >,
         Pair< Vector<     CustomScalar< int >, Devices::Cuda >, VectorView< CustomScalar< double >,       Devices::Cuda > >,
         Pair< Vector<     CustomScalar< int >, Devices::Cuda >, VectorView< const CustomScalar< double >, Devices::Cuda > >,
         Pair< VectorView< CustomScalar< int >, Devices::Cuda >, VectorView< CustomScalar< double >,       Devices::Cuda > >,
         Pair< VectorView< const int,           Devices::Cuda >, VectorView< int,                          Devices::Cuda > >,
         Pair< VectorView< const int,           Devices::Cuda >, VectorView< const int,                    Devices::Cuda > >,
         Pair< VectorView< int,                 Devices::Cuda >, VectorView< const int,                    Devices::Cuda > >
      #endif
      >;
   #endif
#endif

TYPED_TEST_SUITE( VectorBinaryOperationsTest, VectorPairs );

TYPED_TEST( VectorBinaryOperationsTest, EQ )
{
   SETUP_BINARY_TEST_ALIASES;

   EXPECT_EQ( L1, R1 );       // vector or vector view
   EXPECT_EQ( L1, 1 );        // right scalar
   EXPECT_EQ( 1, R1 );        // left scalar
   EXPECT_EQ( L1, RightReal(1) );   // right scalar
   EXPECT_EQ( LeftReal(1), R1 );    // left scalar
   EXPECT_EQ( L2, R1 + R1 );  // right expression
   EXPECT_EQ( L1 + L1, R2 );  // left expression
   EXPECT_EQ( L1 + L1, R1 + R1 );  // two expressions

#ifndef STATIC_VECTOR
   // with different sizes
   EXPECT_FALSE( L1 == Right() );
   // with zero sizes
   EXPECT_TRUE( Left() == Right() );
#endif
}

TYPED_TEST( VectorBinaryOperationsTest, NE )
{
   SETUP_BINARY_TEST_ALIASES;

   EXPECT_NE( L1, R2 );       // vector or vector view
   EXPECT_NE( L1, 2 );        // right scalar
   EXPECT_NE( 2, R1 );        // left scalar
   EXPECT_NE( L1, RightReal(2) );   // right scalar
   EXPECT_NE( LeftReal(2), R1 );    // left scalar
   EXPECT_NE( L1, R1 + R1 );  // right expression
   EXPECT_NE( L1 + L1, R1 );  // left expression
   EXPECT_NE( L1 + L1, R2 + R2 );  // two expressions

#ifndef STATIC_VECTOR
   // with different sizes
   EXPECT_TRUE( L1 != Right() );
   // with zero sizes
   EXPECT_FALSE( Left() != Right() );
#endif
}

TYPED_TEST( VectorBinaryOperationsTest, LT )
{
   SETUP_BINARY_TEST_ALIASES;

   EXPECT_LT( L1, R2 );       // vector or vector view
   EXPECT_LT( L1, 2 );        // right scalar
   EXPECT_LT( 1, R2 );        // left scalar
   EXPECT_LT( L1, RightReal(2) );   // right scalar
   EXPECT_LT( LeftReal(1), R2 );    // left scalar
   EXPECT_LT( L1, R1 + R1 );  // right expression
   EXPECT_LT( L1 - L1, R1 );  // left expression
   EXPECT_LT( L1 - L1, R1 + R1 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, GT )
{
   SETUP_BINARY_TEST_ALIASES;

   EXPECT_GT( L2, R1 );       // vector or vector view
   EXPECT_GT( L2, 1 );        // right scalar
   EXPECT_GT( 2, R1 );        // left scalar
   EXPECT_GT( L2, RightReal(1) );   // right scalar
   EXPECT_GT( LeftReal(2), R1 );    // left scalar
   EXPECT_GT( L1, R1 - R1 );  // right expression
   EXPECT_GT( L1 + L1, R1 );  // left expression
   EXPECT_GT( L1 + L1, R1 - R1 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, LE )
{
   SETUP_BINARY_TEST_ALIASES;

   // same as LT
   EXPECT_LE( L1, R2 );       // vector or vector view
   EXPECT_LE( L1, 2 );        // right scalar
   EXPECT_LE( 1, R2 );        // left scalar
   EXPECT_LE( L1, RightReal(2) );   // right scalar
   EXPECT_LE( LeftReal(1), R2 );    // left scalar
   EXPECT_LE( L1, R1 + R1 );  // right expression
   EXPECT_LE( L1 - L1, R1 );  // left expression
   EXPECT_LE( L1 - L1, R1 + R1 );  // two expressions

   // same as EQ
   EXPECT_LE( L1, R1 );       // vector or vector view
   EXPECT_LE( L1, 1 );        // right scalar
   EXPECT_LE( 1, R1 );        // left scalar
   EXPECT_LE( L1, RightReal(1) );   // right scalar
   EXPECT_LE( LeftReal(1), R1 );    // left scalar
   EXPECT_LE( L2, R1 + R1 );  // right expression
   EXPECT_LE( L1 + L1, R2 );  // left expression
   EXPECT_LE( L1 + L1, R1 + R2 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, GE )
{
   SETUP_BINARY_TEST_ALIASES;

   // same as GT
   EXPECT_GE( L2, R1 );       // vector or vector view
   EXPECT_GE( L2, 1 );        // right scalar
   EXPECT_GE( 2, R1 );        // left scalar
   EXPECT_GE( L2, RightReal(1) );   // right scalar
   EXPECT_GE( LeftReal(2), R1 );    // left scalar
   EXPECT_GE( L1, R1 - R1 );  // right expression
   EXPECT_GE( L1 + L1, R1 );  // left expression
   EXPECT_GE( L1 + L1, R1 - R1 );  // two expressions

   // same as EQ
   EXPECT_LE( L1, R1 );       // vector or vector view
   EXPECT_LE( L1, 1 );        // right scalar
   EXPECT_LE( 1, R1 );        // left scalar
   EXPECT_LE( L1, RightReal(1) );   // right scalar
   EXPECT_LE( LeftReal(1), R1 );    // left scalar
   EXPECT_LE( L2, R1 + R1 );  // right expression
   EXPECT_LE( L1 + L1, R2 );  // left expression
   EXPECT_LE( L1 + L1, R1 + R2 );  // two expressions
}

TYPED_TEST( VectorBinaryOperationsTest, addition )
{
   SETUP_BINARY_TEST_ALIASES;

   // with vector or vector view
   EXPECT_EQ( L1 + R1, 2 );
   // with scalar
   EXPECT_EQ( L1 + 1, 2 );
   EXPECT_EQ( 1 + L1, 2 );
   EXPECT_EQ( L1 + LeftReal(1), 2 );
   EXPECT_EQ( LeftReal(1) + L1, 2 );
   // with expression
   EXPECT_EQ( L1 + (L1 + L1), 3 );
   EXPECT_EQ( (L1 + L1) + L1, 3 );
   EXPECT_EQ( L1 + (L1 + R1), 3 );
   EXPECT_EQ( (L1 + L1) + R1, 3 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) + (L1 + L1), 4 );
   // with expression and scalar
   EXPECT_EQ( (L1 + L1) + 1, 3 );
   EXPECT_EQ( (L1 + L1) + RightReal(1), 3 );
   EXPECT_EQ( 1 + (R1 + R1), 3 );
   EXPECT_EQ( LeftReal(1) + (R1 + R1), 3 );
}

TYPED_TEST( VectorBinaryOperationsTest, subtraction )
{
   SETUP_BINARY_TEST_ALIASES;

   // with vector or vector view
   EXPECT_EQ( L1 - R1, 0 );
   // with scalar
   EXPECT_EQ( L1 - 1, 0 );
   EXPECT_EQ( 1 - L1, 0 );
   EXPECT_EQ( L1 - LeftReal(1), 0 );
   EXPECT_EQ( LeftReal(1) - L1, 0 );
   // with expression
   EXPECT_EQ( L2 - (L1 + L1), 0 );
   EXPECT_EQ( (L1 + L1) - L2, 0 );
   EXPECT_EQ( L2 - (L1 + R1), 0 );
   EXPECT_EQ( (L1 + L1) - R2, 0 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) - (L1 + L1), 0 );
   // with expression and scalar
   EXPECT_EQ( (L1 + L1) - 1, 1 );
   EXPECT_EQ( (L1 + L1) - RightReal(1), 1 );
   EXPECT_EQ( 1 - (R1 + R1), -1 );
   EXPECT_EQ( LeftReal(1) - (R1 + R1), -1 );
}

TYPED_TEST( VectorBinaryOperationsTest, multiplication )
{
   SETUP_BINARY_TEST_ALIASES;

   // with vector or vector view
   EXPECT_EQ( L1 * R2, L2 );
   // with scalar
   EXPECT_EQ( L1 * 2, L2 );
   EXPECT_EQ( 2 * L1, L2 );
   EXPECT_EQ( L1 * LeftReal(2), L2 );
   EXPECT_EQ( LeftReal(2) * L1, L2 );
   // with expression
   EXPECT_EQ( L1 * (L1 + L1), L2 );
   EXPECT_EQ( (L1 + L1) * L1, L2 );
   EXPECT_EQ( L1 * (L1 + R1), L2 );
   EXPECT_EQ( (L1 + L1) * R1, L2 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) * (L1 + L1), 4 );
   // with expression and scalar
   EXPECT_EQ( (L1 + L1) * 1, 2 );
   EXPECT_EQ( (L1 + L1) * RightReal(1), 2 );
   EXPECT_EQ( 1 * (R1 + R1), 2 );
   EXPECT_EQ( LeftReal(1) * (R1 + R1), 2 );
}

TYPED_TEST( VectorBinaryOperationsTest, division )
{
   SETUP_BINARY_TEST_ALIASES;

   // with vector or vector view
   EXPECT_EQ( L2 / R2, L1 );
   // with scalar
   EXPECT_EQ( L2 / 2, L1 );
   EXPECT_EQ( 2 / L2, L1 );
   EXPECT_EQ( L2 / LeftReal(2), L1 );
   EXPECT_EQ( LeftReal(2) / L2, L1 );
   // with expression
   EXPECT_EQ( L2 / (L1 + L1), L1 );
   EXPECT_EQ( (L1 + L1) / L2, L1 );
   EXPECT_EQ( L2 / (L1 + R1), L1 );
   EXPECT_EQ( (L1 + L1) / R2, L1 );
   // with two expressions
   EXPECT_EQ( (L1 + L1) / (L1 + L1), L1 );
   // with expression and scalar
   EXPECT_EQ( (L1 + L1) / 1, 2 );
   EXPECT_EQ( (L1 + L1) / RightReal(1), 2 );
   EXPECT_EQ( 2 / (R1 + R1), 1 );
   EXPECT_EQ( LeftReal(2) / (R1 + R1), 1 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   using RightReal = std::remove_const_t< typename Right::RealType >;
   // with vector or vector view
   L1 = R2;
   EXPECT_EQ( L1, R2 );
   // with scalar
   L1 = 1;
   EXPECT_EQ( L1, 1 );
   L1 = RightReal(1);
   EXPECT_EQ( L1, 1 );
   // with expression
   L1 = R1 + R1;
   EXPECT_EQ( L1, R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, assignment )
{
   SETUP_BINARY_TEST_ALIASES;
   test_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_add_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_add_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   using RightReal = std::remove_const_t< typename Right::RealType >;
   // with vector or vector view
   L1 += R2;
   EXPECT_EQ( L1, R1 + R2 );
   // with scalar
   L1 = 1;
   L1 += 2;
   EXPECT_EQ( L1, 3 );
   L1 = 1;
   L1 += RightReal(2);
   EXPECT_EQ( L1, 3 );
   // with expression
   L1 = 1;
   L1 += R1 + R1;
   EXPECT_EQ( L1, R1 + R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, add_assignment )
{
   SETUP_BINARY_TEST_ALIASES;
   test_add_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_subtract_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_subtract_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   using RightReal = std::remove_const_t< typename Right::RealType >;
   // with vector or vector view
   L1 -= R2;
   EXPECT_EQ( L1, R1 - R2 );
   // with scalar
   L1 = 1;
   L1 -= 2;
   EXPECT_EQ( L1, -1 );
   L1 = 1;
   L1 -= RightReal(2);
   EXPECT_EQ( L1, -1 );
   // with expression
   L1 = 1;
   L1 -= R1 + R1;
   EXPECT_EQ( L1, -R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, subtract_assignment )
{
   SETUP_BINARY_TEST_ALIASES;
   test_subtract_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_multiply_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_multiply_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   using RightReal = std::remove_const_t< typename Right::RealType >;
   // with vector or vector view
   L1 *= R2;
   EXPECT_EQ( L1, R2 );
   // with scalar
   L1 = 1;
   L1 *= 2;
   EXPECT_EQ( L1, 2 );
   L1 = 1;
   L1 *= RightReal(2);
   EXPECT_EQ( L1, 2 );
   // with expression
   L1 = 1;
   L1 *= R1 + R1;
   EXPECT_EQ( L1, R1 + R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, multiply_assignment )
{
   SETUP_BINARY_TEST_ALIASES;
   test_multiply_assignment( L1, L2, R1, R2 );
}

template< typename Left, typename Right, std::enable_if_t< std::is_const<typename Left::RealType>::value, bool > = true >
void test_divide_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{}
template< typename Left, typename Right, std::enable_if_t< ! std::is_const<typename Left::RealType>::value, bool > = true >
void test_divide_assignment( Left& L1, Left& L2, Right& R1, Right& R2 )
{
   using RightReal = std::remove_const_t< typename Right::RealType >;
   // with vector or vector view
   L2 /= R2;
   EXPECT_EQ( L1, R1 );
   // with scalar
   L2 = 2;
   L2 /= 2;
   EXPECT_EQ( L1, 1 );
   L1 = 2;
   L1 /= RightReal(2);
   EXPECT_EQ( L1, 1 );
   // with expression
   L2 = 2;
   L2 /= R1 + R1;
   EXPECT_EQ( L1, R1 );
}
TYPED_TEST( VectorBinaryOperationsTest, divide_assignment )
{
   SETUP_BINARY_TEST_ALIASES;
   test_divide_assignment( L1, L2, R1, R2 );
}

TYPED_TEST( VectorBinaryOperationsTest, scalarProduct )
{
   this->reset( VECTOR_TEST_REDUCTION_SIZE );

#ifdef STATIC_VECTOR
   setOscilatingSequence( this->L1, 1 );
   setConstantSequence( this->R1, 1 );

   const typename TestFixture::Left& L( this->L1 );
   const typename TestFixture::Right& R( this->R1 );
#else
   // we have to use _L1 and _R1 because L1 and R1 might be a const view
   setOscilatingSequence( this->_L1, 1 );
   setConstantSequence( this->_R1, 1 );

   const typename TestFixture::Left L( this->_L1 );
   const typename TestFixture::Right R( this->_R1 );
#endif

   const int size = L.getSize();
   const int expected = size % 2 ? 1 : 0;

   // vector or vector view
   EXPECT_EQ( dot(L, R), expected );
   EXPECT_EQ( (L, R), expected );
   // left expression
   EXPECT_EQ( dot(2 * L - L, R), expected );
   EXPECT_EQ( (2 * L - L, R), expected );
   // right expression
   EXPECT_EQ( dot(L, 2 * R - R), expected );
   EXPECT_EQ( (L, 2 * R - R), expected );
   // both expressions
   EXPECT_EQ( dot(2 * L - L, 2 * R - R), expected );
   EXPECT_EQ( (2 * L - L, 2 * R - R), expected );
}

TYPED_TEST( VectorBinaryOperationsTest, min )
{
   SETUP_BINARY_TEST_ALIASES;

   // make sure we are testing TNL::min and not std::min (which might take precedence when
   // both arguments have the same type, see https://stackoverflow.com/q/62847698/4180822
   using TNL::min;

   // with vector or vector view
   EXPECT_EQ( min(L1, R2), L1 );
   // with scalar
   EXPECT_EQ( min(L1, 2), L1 );
   EXPECT_EQ( min(1, R2), L1 );
   // with expression
   EXPECT_EQ( min(L1, R1 + R1), L1 );
   EXPECT_EQ( min(L1 + L1, R1), R1 );
   // with two expressions
   EXPECT_EQ( min(L1 + L1, R1 + R2), L2 );
   // with expression and scalar
   EXPECT_EQ( min(L1 + L1, 1), L1 );
   EXPECT_EQ( min(L1 + L1, RightReal(1)), L1 );
   EXPECT_EQ( min(1, R1 + R1), L1 );
   EXPECT_EQ( min(LeftReal(1), R1 + R1), L1 );
}

TYPED_TEST( VectorBinaryOperationsTest, max )
{
   SETUP_BINARY_TEST_ALIASES;

   // make sure we are testing TNL::max and not std::max (which might take precedence when
   // both arguments have the same type, see https://stackoverflow.com/q/62847698/4180822
   using TNL::max;

   // with vector or vector view
   EXPECT_EQ( max(L1, R2), R2 );
   // with scalar
   EXPECT_EQ( max(L1, 2), L2 );
   EXPECT_EQ( max(1, R2), L2 );
   // with expression
   EXPECT_EQ( max(L1, R1 + R1), L2 );
   EXPECT_EQ( max(L1 + L1, R1), R2 );
   // with two expressions
   EXPECT_EQ( max(L1 - L1, R1 + R1), L2 );
   // with expression and scalar
   EXPECT_EQ( max(L1 + L1, 1), L2 );
   EXPECT_EQ( max(L1 + L1, RightReal(1)), L2 );
   EXPECT_EQ( max(1, R1 + R1), L2 );
   EXPECT_EQ( max(LeftReal(1), R1 + R1), L2 );
}

#if defined(__CUDACC__) && !defined(STATIC_VECTOR)
TYPED_TEST( VectorBinaryOperationsTest, comparisonOnDifferentDevices )
{
   SETUP_BINARY_TEST_ALIASES;

   using RightHostVector = typename TestFixture::RightVector::template Self< typename TestFixture::RightVector::RealType, Devices::Sequential >;
   using RightHost = typename TestFixture::Right::template Self< typename TestFixture::Right::RealType, Devices::Sequential >;

   RightHostVector _R1_h; _R1_h = this->_R1;
   RightHost R1_h( _R1_h );

   // L1 and L2 are device vectors
   EXPECT_EQ( L1, R1_h );
   EXPECT_NE( L2, R1_h );
}
#endif

#ifdef __CUDACC__
   #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
      #pragma nv_diagnostic pop
   #else
      #pragma pop
   #endif
#endif

} // namespace binary_tests

#endif // HAVE_GTEST
