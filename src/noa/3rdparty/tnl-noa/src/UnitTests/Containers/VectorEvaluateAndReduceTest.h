#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorTestSetup.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int VECTOR_TEST_SIZE = 500;

// NOTE: The following lambdas cannot be inside the test because of nvcc ( v. 10.1.105 )
// error #3049-D: The enclosing parent function ("TestBody") for an extended __host__ __device__ lambda cannot have private or protected access within its class
template< typename VectorView >
typename VectorView::RealType
performEvaluateAndReduce( VectorView& u, VectorView& v, VectorView& w )
{
   using RealType = typename VectorView::RealType;

   return evaluateAndReduce( w, u * v, std::plus<>{}, ( RealType ) 0.0 );
}

TYPED_TEST( VectorTest, evaluateAndReduce )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   RealType aux( 0.0 );
   for( int i = 0; i < size; i++ )
   {
      const RealType x = i;
      const RealType y = size / 2 - i;
      u.setElement( i, x );
      v.setElement( i, y );
      aux += x * y;
   }
   auto r = performEvaluateAndReduce( u, v, w );
   EXPECT_TRUE( w == u * v );
   EXPECT_NEAR( aux, r, 1.0e-5 );
}

// NOTE: The following lambdas cannot be inside the test because of nvcc ( v. 10.1.105 )
// error #3049-D: The enclosing parent function ("TestBody") for an extended __host__ __device__ lambda cannot have private or protected access within its class
template< typename VectorView >
typename VectorView::RealType
performAddAndReduce1( VectorView& u, VectorView& v, VectorView& w )
{
   using RealType = typename VectorView::RealType;

   return addAndReduce( w, u * v, std::plus<>{}, ( RealType ) 0.0 );
}

template< typename VectorView >
typename VectorView::RealType
performAddAndReduce2( VectorView& v, VectorView& w )
{
   using RealType = typename VectorView::RealType;

   return addAndReduce( w, 5.0 * v, std::plus<>{}, ( RealType ) 0.0 );
}


TYPED_TEST( VectorTest, addAndReduce )
{
   using VectorType = typename TestFixture::VectorType;
   using ViewType = typename TestFixture::ViewType;
   using RealType = typename VectorType::RealType;
   using IndexType = typename VectorType::IndexType;
   const int size = VECTOR_TEST_SIZE;

   VectorType _u( size ), _v( size ), _w( size );
   ViewType u( _u ), v( _v ), w( _w );
   RealType aux( 0.0 );
   for( int i = 0; i < size; i++ )
   {
      const RealType x = i;
      const RealType y = size / 2 - i;
      u.setElement( i, x );
      v.setElement( i, y );
      w.setElement( i, x );
      aux += x * y;
   }
   auto r = performAddAndReduce1( u, v, w );
   EXPECT_TRUE( w == u * v + u );
   EXPECT_NEAR( aux, r, 1.0e-5 );

   aux = 0.0;
   for( int i = 0; i < size; i++ )
   {
      const RealType x = i;
      const RealType y = size / 2 - i;
      u.setElement( i, x );
      v.setElement( i, y );
      w.setElement( i, x );
      aux += 5.0 * y;
   }
   r = performAddAndReduce2( v, w );
   EXPECT_TRUE( w == 5.0 * v + u );
   EXPECT_NEAR( aux, r, 1.0e-5 );

}

#endif // HAVE_GTEST

#include "../main.h"
