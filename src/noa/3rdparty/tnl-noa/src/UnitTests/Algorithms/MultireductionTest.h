#pragma once

#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Multireduction.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename View >
void setLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
   deviceVector = a;
}

template< typename View >
void setNegativeLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
   deviceVector = a;
}

// test fixture for typed tests
template< typename Vector >
class MultireductionTest : public ::testing::Test
{
protected:
   using DeviceVector = Vector;
   using DeviceView = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
   using HostVector = typename DeviceVector::template Self< typename DeviceVector::RealType, Devices::Sequential >;
   using HostView = typename DeviceView::template Self< typename DeviceView::RealType, Devices::Sequential >;

   // should be small enough to have fast tests, but larger than minGPUReductionDataSize
   // and large enough to require multiple CUDA blocks for reduction
   static constexpr int size = 5000;

   // number of vectors which are reduced together
   static constexpr int n = 4;

   DeviceVector V;
   DeviceVector y;
   HostVector result;

   MultireductionTest()
   {
      V.setSize( size * n );
      y.setSize( size );
      result.setSize( n );

      for( int i = 0; i < n; i++ ) {
         DeviceView v( V.getData() + i * size, size );
         if( i % 2 == 0 )
            setLinearSequence( v );
         else
            setNegativeLinearSequence( v );
      }
      y.setValue( 1 );
   }
};

// types for which MultireductionTest is instantiated
using VectorTypes = ::testing::Types<
   Vector< int,   Devices::Host >,
   Vector< float, Devices::Host >
#ifdef HAVE_CUDA
   ,
   Vector< int,   Devices::Cuda >,
   Vector< float, Devices::Cuda >
#endif
>;

TYPED_TEST_SUITE( MultireductionTest, VectorTypes );

// idiot nvcc does not allow __cuda_callable__ lambdas inside private or protected regions
template< typename DeviceVector, typename HostVector >
void test_multireduction( const DeviceVector& V, const DeviceVector& y, HostVector& result )
{
   using RealType = typename DeviceVector::RealType;
   using DeviceType = typename DeviceVector::DeviceType;
   using IndexType = typename DeviceVector::IndexType;

   const RealType* _V = V.getData();
   const RealType* _y = y.getData();
   const IndexType size = y.getSize();
   const int n = result.getSize();
   ASSERT_EQ( V.getSize(), size * n );

   auto fetch = [=] __cuda_callable__ ( IndexType i, int k )
   {
      TNL_ASSERT_LT( i, size, "BUG: fetcher got invalid index i" );
      TNL_ASSERT_LT( k, n, "BUG: fetcher got invalid index k" );
      return _V[ i + k * size ] * _y[ i ];
   };
   Multireduction< DeviceType >::reduce
               ( (RealType) 0,
                 fetch,
                 std::plus<>{},
                 size,
                 n,
                 result.getData() );

   for( int i = 0; i < n; i++ ) {
      if( i % 2 == 0 )
         EXPECT_EQ( result[ i ], 0.5 * size * ( size - 1 ) );
      else
         EXPECT_EQ( result[ i ], - 0.5 * size * ( size - 1 ) );
   }
}
TYPED_TEST( MultireductionTest, scalarProduct )
{
   test_multireduction( this->V, this->y, this->result );
}
#endif // HAVE_GTEST


#include "../main.h"
