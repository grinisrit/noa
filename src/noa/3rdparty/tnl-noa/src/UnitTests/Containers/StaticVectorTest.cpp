#ifdef HAVE_GTEST
#include <TNL/Containers/StaticVector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// test fixture for typed tests
template< typename Vector >
class StaticVectorTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using RealType = typename VectorType::RealType;
};

// types for which VectorTest is instantiated
using StaticVectorTypes = ::testing::Types<
   StaticVector< 1, short >,
   StaticVector< 1, int >,
   StaticVector< 1, long >,
   StaticVector< 1, float >,
   StaticVector< 1, double >,
   StaticVector< 2, short >,
   StaticVector< 2, int >,
   StaticVector< 2, long >,
   StaticVector< 2, float >,
   StaticVector< 2, double >,
   StaticVector< 3, short >,
   StaticVector< 3, int >,
   StaticVector< 3, long >,
   StaticVector< 3, float >,
   StaticVector< 3, double >,
   StaticVector< 4, short >,
   StaticVector< 4, int >,
   StaticVector< 4, long >,
   StaticVector< 4, float >,
   StaticVector< 4, double >,
   StaticVector< 5, short >,
   StaticVector< 5, int >,
   StaticVector< 5, long >,
   StaticVector< 5, float >,
   StaticVector< 5, double >
>;

TYPED_TEST_SUITE( StaticVectorTest, StaticVectorTypes );


TYPED_TEST( StaticVectorTest, constructors )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   constexpr int size = VectorType::getSize();

   RealType data[ size ];
   for( int i = 0; i < size; i++ )
      data[ i ] = i;

   VectorType u0;
   EXPECT_TRUE( u0.getData() );

   VectorType u1( data );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u1[ i ], data[ i ] );

   VectorType u2( 7 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u2[ i ], 7 );

   VectorType u3( u1 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u3[ i ], u1[ i ] );

   // initialization with 0 requires special treatment to avoid ambiguity,
   // see https://stackoverflow.com/q/4610503
   VectorType v( 0 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v[ i ], 0 );
}

TYPED_TEST( StaticVectorTest, cast )
{
   using VectorType = typename TestFixture::VectorType;
   constexpr int size = VectorType::getSize();

   VectorType u( 1 );
   EXPECT_EQ( (StaticVector< size, double >) u, u );
}

#endif

#include "../main.h"
