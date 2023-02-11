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

namespace vertical_tests {

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_REDUCTION_SIZE = 4999;

// test fixture for typed tests
template< typename T >
class VectorVerticalOperationsTest : public ::testing::Test
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

      // some arbitrary value (but must be 0 if not distributed)
      const int ghosts = (nproc > 1) ? 4 : 0;
   #else
      using VectorType = Containers::Vector< NonConstReal, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
      template< typename Real >
      using Vector = Containers::Vector< Real, typename VectorOrView::DeviceType, typename VectorOrView::IndexType >;
   #endif
#endif

   VectorOrView V1;

#ifndef STATIC_VECTOR
   VectorType _V1;
#endif

   void reset( int size )
   {
#ifdef STATIC_VECTOR
      setLinearSequence( V1 );
#else
   #ifdef DISTRIBUTED_VECTOR
      using LocalRangeType = typename VectorOrView::LocalRangeType;
      using Synchronizer = typename Partitioner< typename VectorOrView::IndexType >::template ArraySynchronizer< typename VectorOrView::DeviceType >;
      const LocalRangeType localRange = Partitioner< typename VectorOrView::IndexType >::splitRange( size, communicator );
      _V1.setDistribution( localRange, ghosts, size, communicator );
      _V1.setSynchronizer( std::make_shared<Synchronizer>( localRange, ghosts / 2, communicator ) );
   #else
      _V1.setSize( size );
   #endif
      setLinearSequence( _V1 );
      bindOrAssign( V1, _V1 );
#endif
   }

   VectorVerticalOperationsTest()
   {
      reset( VECTOR_TEST_REDUCTION_SIZE );
   }
};

#define SETUP_VERTICAL_TEST_ALIASES \
   using VectorOrView = typename TestFixture::VectorOrView; \
   VectorOrView& V1 = this->V1;                             \
   const int size = V1.getSize();                           \
   (void) 0  // dummy statement here enforces ';' after the macro use

// types for which VectorVerticalOperationsTest is instantiated
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
         VectorView< StaticVector< 3, double >, Devices::Host >
      #else
         Vector<     StaticVector< 3, double >, Devices::Cuda >,
         VectorView< StaticVector< 3, double >, Devices::Cuda >
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

TYPED_TEST_SUITE( VectorVerticalOperationsTest, VectorTypes );

// FIXME: function does not work for nested vectors - std::numeric_limits does not make sense for vector types
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, max )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( max(V1), size - 1 );
   // unary expression
   EXPECT_EQ( max(-V1), 0 );
   // binary expression
   EXPECT_EQ( max(V1 + 2), size - 1 + 2 );
}
#endif

// FIXME: function does not work for nested vectors - the reduction operation expects a scalar type
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, argMax )
{
   SETUP_VERTICAL_TEST_ALIASES;
   using RealType = typename TestFixture::VectorOrView::RealType;

   // vector or view
   EXPECT_EQ( argMax(V1), std::make_pair( (RealType) size - 1 , size - 1 ) );
   // unary expression
   EXPECT_EQ( argMax(-V1), std::make_pair( (RealType) 0, 0 ) );
   // expression
   EXPECT_EQ( argMax(V1 + 2), std::make_pair( (RealType) size - 1 + 2, size - 1 ) );
}
#endif

// FIXME: function does not work for nested vectors - std::numeric_limits does not make sense for vector types
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, min )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( min(V1), 0 );
   // unary expression
   EXPECT_EQ( min(-V1), 1 - size );
   // binary expression
   EXPECT_EQ( min(V1 + 2), 2 );
}
#endif

// FIXME: function does not work for nested vectors - the reduction operation expects a scalar type
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, argMin )
{
   SETUP_VERTICAL_TEST_ALIASES;
   using RealType = typename TestFixture::VectorOrView::RealType;

   // vector or view
   EXPECT_EQ( argMin(V1), std::make_pair( (RealType) 0, 0 ) );
   // unary expression
   EXPECT_EQ( argMin(-V1), std::make_pair( (RealType) 1 - size, size - 1 ) );
   // binary expression
   EXPECT_EQ( argMin(V1 + 2), std::make_pair( (RealType) 2 , 0 ) );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, sum )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( sum(V1), 0.5 * size * (size - 1) );
   // unary expression
   EXPECT_EQ( sum(-V1), - 0.5 * size * (size - 1) );
   // binary expression
   EXPECT_EQ( sum(V1 - 1), 0.5 * size * (size - 1) - size );
}

// FIXME: function does not work for nested vectors - max does not work for nested vectors
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, maxNorm )
{
   SETUP_VERTICAL_TEST_ALIASES;

   // vector or view
   EXPECT_EQ( maxNorm(V1), size - 1 );
   // unary expression
   EXPECT_EQ( maxNorm(-V1), size - 1 );
   // binary expression
   EXPECT_EQ( maxNorm(V1 - size), size );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, l1Norm )
{
#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
#endif
   const int size = V1.getSize();

   // vector or vector view
   EXPECT_EQ( l1Norm(V1), size );
   // unary expression
   EXPECT_EQ( l1Norm(-V1), size );
   // binary expression
   EXPECT_EQ( l1Norm(2 * V1 - V1), size );
}

// FIXME: l2Norm does not work for nested vectors - dangling references due to Static*ExpressionTemplate
//        classes binding to temporary objects which get destroyed before l2Norm returns
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, l2Norm )
{
#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
#endif
   const int size = V1.getSize();

   const auto expected = std::sqrt( size );

   // vector or vector view
   EXPECT_EQ( l2Norm(V1), expected );
   // unary expression
   EXPECT_EQ( l2Norm(-V1), expected );
   // binary expression
   EXPECT_EQ( l2Norm(2 * V1 - V1), expected );
}
#endif

// FIXME function does not work for nested vectors - compilation error
#ifndef VECTOR_OF_STATIC_VECTORS
TYPED_TEST( VectorVerticalOperationsTest, lpNorm )
{
#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 1 );
   const typename TestFixture::VectorOrView& V1( this->_V1 );
#endif
   const int size = V1.getSize();

   const auto expectedL1norm = size;
   const auto expectedL2norm = std::sqrt( size );
   const auto expectedL3norm = std::cbrt( size );

   const auto epsilon = 64 * std::numeric_limits< decltype(expectedL3norm) >::epsilon();

   // vector or vector view
   EXPECT_EQ( lpNorm(V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(V1, 2.0), expectedL2norm );
   expect_near( lpNorm(V1, 3.0), expectedL3norm, epsilon );
   // unary expression
   EXPECT_EQ( lpNorm(-V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(-V1, 2.0), expectedL2norm );
   expect_near( lpNorm(-V1, 3.0), expectedL3norm, epsilon );
   // binary expression
   EXPECT_EQ( lpNorm(2 * V1 - V1, 1.0), expectedL1norm );
   EXPECT_EQ( lpNorm(2 * V1 - V1, 2.0), expectedL2norm );
   expect_near( lpNorm(2 * V1 - V1, 3.0), expectedL3norm, epsilon );
}
#endif

TYPED_TEST( VectorVerticalOperationsTest, product )
{
   // VERY small size to avoid overflows
   this->reset( 16 );

#ifdef STATIC_VECTOR
   setConstantSequence( this->V1, 2 );
   const typename TestFixture::VectorOrView& V2( this->V1 );
#else
   // we have to use _V1 because V1 might be a const view
   setConstantSequence( this->_V1, 2 );
   const typename TestFixture::VectorOrView& V2( this->_V1 );
#endif
   const int size = V2.getSize();

   // vector or vector view
   EXPECT_EQ( product(V2), std::exp2(size) );
   // unary expression
   EXPECT_EQ( product(-V2), std::exp2(size) * ( (size % 2) ? -1 : 1 ) );
   // binary expression
   EXPECT_EQ( product(2 * V2 - V2), std::exp2(size) );
}

// TODO: tests for logicalOr, binaryOr, logicalAnd, binaryAnd

} // namespace vertical_tests

#endif // HAVE_GTEST
