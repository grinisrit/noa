#pragma once

#ifdef HAVE_GTEST
#include "VectorTestSetup.h"

constexpr int VECTOR_TEST_SIZE = 100;

TYPED_TEST( VectorTest, constructors )
{
   using VectorType = typename TestFixture::VectorType;
   const int size = VECTOR_TEST_SIZE;

   VectorType empty_u;
   VectorType empty_v( empty_u );
   EXPECT_EQ( empty_u.getSize(), 0 );
   EXPECT_EQ( empty_v.getSize(), 0 );

   VectorType u( size );
   EXPECT_EQ( u.getSize(), size );

   VectorType v( 10 );
   EXPECT_EQ( v.getSize(), 10 );

   if( std::is_same< typename VectorType::DeviceType, Devices::Host >::value ) {
      typename VectorType::ValueType data[ 10 ];
      VectorType w( data, 10 );
      EXPECT_NE( w.getData(), data );

      VectorType z1( w );
      EXPECT_NE( z1.getData(), data );
      EXPECT_EQ( z1.getSize(), 10 );

      VectorType z2( w, 1 );
      EXPECT_EQ( z2.getSize(), 9 );

      VectorType z3( w, 2, 3 );
      EXPECT_EQ( z3.getSize(), 3 );
   }

   v = 1;
   VectorType w( v );
   EXPECT_EQ( w.getSize(), v.getSize() );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), w.getElement( i ) );
   v.reset();
   EXPECT_EQ( w.getSize(), 10 );

   VectorType a1 { 1, 2, 3 };
   EXPECT_EQ( a1.getElement( 0 ), 1 );
   EXPECT_EQ( a1.getElement( 1 ), 2 );
   EXPECT_EQ( a1.getElement( 2 ), 3 );

   std::list< int > l = { 4, 5, 6 };
   VectorType a2( l );
   EXPECT_EQ( a2.getElement( 0 ), 4 );
   EXPECT_EQ( a2.getElement( 1 ), 5 );
   EXPECT_EQ( a2.getElement( 2 ), 6 );

   std::vector< int > q = { 7, 8, 9 };

   VectorType a3( q );
   EXPECT_EQ( a3.getElement( 0 ), 7 );
   EXPECT_EQ( a3.getElement( 1 ), 8 );
   EXPECT_EQ( a3.getElement( 2 ), 9 );

   VectorType a4( 2 * a2 + 3 * a3 );
   EXPECT_EQ( a4.getElement( 0 ), 2.0 * a2.getElement( 0 ) + 3 * a3.getElement( 0 ) );
   EXPECT_EQ( a4.getElement( 1 ), 2.0 * a2.getElement( 1 ) + 3 * a3.getElement( 1 ) );
   EXPECT_EQ( a4.getElement( 2 ), 2.0 * a2.getElement( 2 ) + 3 * a3.getElement( 2 ) );

}

TEST( VectorSpecialCasesTest, defaultConstructors )
{
   #ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
   #else
   using DeviceType = TNL::Devices::Host;
   #endif
   using ArrayType = Containers::Array< int, DeviceType >;
   using VectorViewType = VectorView< int, DeviceType >;
   using ArrayViewType = ArrayView< int, DeviceType >;

   ArrayType a( 100 );
   a.setValue( 0 );

   ArrayViewType a_view;
   a_view.bind( a );

   VectorViewType v_view;
   v_view.bind( a );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
}

TEST( VectorSpecialCasesTest, assignmentThroughView )
{
   #ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
   #else
   using DeviceType = TNL::Devices::Host;
   #endif
   using VectorType = Containers::Vector< int, DeviceType >;
   using ViewType = VectorView< int, DeviceType >;

   static_assert( HasSubscriptOperator< VectorType >::value, "Subscript operator detection by SFINAE does not work for Vector." );
   static_assert( HasSubscriptOperator< ViewType >::value, "Subscript operator detection by SFINAE does not work for VectorView." );

   VectorType u( 100 ), v( 100 );
   ViewType u_view( u ), v_view( v );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u_view;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view.getElement( i ), 42 );

   u.setValue( 42 );
   v.setValue( 0 );
   v_view = u;
   EXPECT_EQ( u_view.getData(), u.getData() );
   EXPECT_EQ( v_view.getData(), v.getData() );
   for( int i = 0; i < 100; i++ )
      EXPECT_EQ( v_view.getElement( i ), 42 );
}

TEST( VectorSpecialCasesTest, initializationOfVectorViewByArrayView )
{
   #ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
   #else
   using DeviceType = TNL::Devices::Host;
   #endif
   using ArrayType = Containers::Array< int, DeviceType >;
   using VectorViewType = VectorView< const int, DeviceType >;
   using ArrayViewType = ArrayView< int, DeviceType >;

   ArrayType a( 100 );
   a.setValue( 0 );
   ArrayViewType a_view( a );

   VectorViewType v_view( a_view );
   EXPECT_EQ( v_view.getData(), a_view.getData() );
   EXPECT_EQ( sum( v_view ), 0 );
}

TEST( VectorSpecialCasesTest, sumOfBoolVector )
{
   #ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
   #else
   using DeviceType = TNL::Devices::Host;
   #endif
   using VectorType = Containers::Vector< bool, DeviceType >;
   using ViewType = typename VectorType::ViewType;
   const double epsilon = 64 * std::numeric_limits< double >::epsilon();
   constexpr int size = 4999;

   VectorType v( size );
   ViewType v_view( v );
   v.setValue( true );

   // normal sum and lpNorm rely on built-in integral promotion
   const auto sum = TNL::sum( v );
   const auto l1norm = l1Norm( v );
   const auto l2norm = l2Norm( v );
   const auto l3norm = lpNorm( v, 3.0 );
   EXPECT_EQ( sum, size );
   EXPECT_EQ( l1norm, size );
   EXPECT_EQ( l2norm, std::sqrt( size ) );
   EXPECT_NEAR( l3norm, std::cbrt( size ), epsilon );

   // explicit cast to double
   const auto sum_cast = TNL::sum( cast<double>( v ) );
   const auto l1norm_cast = l1Norm( cast<double>( v ) );
   const auto l2norm_cast = l2Norm( cast<double>( v ) );
   const auto l3norm_cast = lpNorm( cast<double>( v ), 3.0 );
   EXPECT_EQ( sum_cast, size );
   EXPECT_EQ( l1norm_cast, size );
   EXPECT_EQ( l2norm_cast, std::sqrt( size ) );
   EXPECT_NEAR( l3norm_cast, std::cbrt( size ), epsilon );

   // test views
   // normal sum and lpNorm rely on built-in integral promotion
   const auto sum_view = TNL::sum( v_view );
   const auto l1norm_view = l1Norm( v_view );
   const auto l2norm_view = l2Norm( v_view );
   const auto l3norm_view = lpNorm( v_view, 3.0 );
   EXPECT_EQ( sum_view, size );
   EXPECT_EQ( l1norm_view, size );
   EXPECT_EQ( l2norm_view, std::sqrt( size ) );
   EXPECT_NEAR( l3norm_view, std::cbrt( size ), epsilon );

   // explicit cast to double
   const auto sum_view_cast = TNL::sum( cast<double>( v_view ) );
   const auto l1norm_view_cast = l1Norm( cast<double>( v_view ) );
   const auto l2norm_view_cast = l2Norm( cast<double>( v_view ) );
   const auto l3norm_view_cast = lpNorm( cast<double>( v_view ), 3.0 );
   EXPECT_EQ( sum_view_cast, size );
   EXPECT_EQ( l1norm_view_cast, size);
   EXPECT_EQ( l2norm_view_cast, std::sqrt( size ) );
   EXPECT_NEAR( l3norm_view_cast, std::cbrt( size ), epsilon );
}

TEST( VectorSpecialCasesTest, reductionOfEmptyVector )
{
   #ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
   #else
   using DeviceType = TNL::Devices::Host;
   #endif
   using VectorType = Containers::Vector< int, DeviceType >;
   using ViewType = typename VectorType::ViewType;
   constexpr int size = 0;

   VectorType v( size );
   ViewType v_view( v );
   v.setValue( 1 );

   EXPECT_EQ( min(v), std::numeric_limits< int >::max() );
   EXPECT_EQ( max(v), std::numeric_limits< int >::lowest() );
   EXPECT_EQ( argMin(v), std::make_pair( std::numeric_limits< int >::max(), 0 ) );
   EXPECT_EQ( argMax(v), std::make_pair( std::numeric_limits< int >::lowest(), 0 ) );
   EXPECT_EQ( sum(v), 0 );
   EXPECT_EQ( product(v), 1 );
   EXPECT_EQ( logicalAnd(v), true );
   EXPECT_EQ( logicalOr(v), false );
   EXPECT_EQ( binaryAnd(v), ~0 );
   EXPECT_EQ( binaryOr(v), 0 );

   EXPECT_EQ( min(v_view), std::numeric_limits< int >::max() );
   EXPECT_EQ( max(v_view), std::numeric_limits< int >::lowest() );
   EXPECT_EQ( argMin(v_view), std::make_pair( std::numeric_limits< int >::max(), 0 ) );
   EXPECT_EQ( argMax(v_view), std::make_pair( std::numeric_limits< int >::lowest(), 0 ) );
   EXPECT_EQ( sum(v_view), 0 );
   EXPECT_EQ( product(v_view), 1 );
   EXPECT_EQ( logicalAnd(v_view), true );
   EXPECT_EQ( logicalOr(v_view), false );
   EXPECT_EQ( binaryAnd(v_view), ~0 );
   EXPECT_EQ( binaryOr(v_view), 0 );
}

#endif // HAVE_GTEST

#include "../main.h"
