#pragma once

#include <TNL/Algorithms/reduce.h>
#include "CommonVectorOperations.h"

namespace TNL {
namespace Benchmarks {

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorMax( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::max( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorMin( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::min( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorAbsMax( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data[ i ] ); };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::max( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorAbsMin( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data[ i ] ); };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::min( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorL1Norm( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data[ i ] ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorL2Norm( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data[ i ] * data[ i ]; };
   return std::sqrt( Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 ) );
}

template< typename Device >
   template< typename Vector, typename ResultType, typename Scalar >
ResultType
CommonVectorOperations< Device >::
getVectorLpNorm( const Vector& v,
                 const Scalar p )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   using IndexType = typename Vector::IndexType;

   if( p == 1.0 )
      return getVectorL1Norm< Vector, ResultType >( v );
   if( p == 2.0 )
      return getVectorL2Norm< Vector, ResultType >( v );

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::pow( TNL::abs( data[ i ] ), p ); };
   return std::pow( Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 ), 1.0 / p );
}

template< typename Device >
   template< typename Vector, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorSum( const Vector& v )
{
   TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );

   if( std::is_same< ResultType, bool >::value )
      abort();

   using IndexType = typename Vector::IndexType;

   const auto* data = v.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i )  -> ResultType { return data[ i ]; };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceMax( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data1[ i ] - data2[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::max( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceMin( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data1[ i ] - data2[ i ]; };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::min( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceAbsMax( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data1[ i ] - data2[ i ] ); };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::max( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(), fetch, reduction, std::numeric_limits< ResultType >::lowest() );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceAbsMin( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data1[ i ] - data2[ i ] ); };
   auto reduction = [] __cuda_callable__ ( const ResultType& a, const ResultType& b ) { return TNL::min( a, b ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(), fetch, reduction, std::numeric_limits< ResultType >::max() );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceL1Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::abs( data1[ i ] - data2[ i ] ); };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceL2Norm( const Vector1& v1,
                           const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) {
      auto diff = data1[ i ] - data2[ i ];
      return diff * diff;
   };
   return std::sqrt( Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 ) );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType, typename Scalar >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceLpNorm( const Vector1& v1,
                           const Vector2& v2,
                           const Scalar p )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );
   TNL_ASSERT_GE( p, 1.0, "Parameter of the L^p norm must be at least 1.0." );

   if( p == 1.0 )
      return getVectorDifferenceL1Norm< Vector1, Vector2, ResultType >( v1, v2 );
   if( p == 2.0 )
      return getVectorDifferenceL2Norm< Vector1, Vector2, ResultType >( v1, v2 );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return TNL::pow( TNL::abs( data1[ i ] - data2[ i ] ), p ); };
   return std::pow( Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 ), 1.0 / p );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getVectorDifferenceSum( const Vector1& v1,
                        const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data1[ i ] - data2[ i ]; };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 );
}

template< typename Device >
   template< typename Vector1, typename Vector2, typename ResultType >
ResultType
CommonVectorOperations< Device >::
getScalarProduct( const Vector1& v1,
                  const Vector2& v2 )
{
   TNL_ASSERT_GT( v1.getSize(), 0, "Vector size must be positive." );
   TNL_ASSERT_EQ( v1.getSize(), v2.getSize(), "The vector sizes must be the same." );

   using IndexType = typename Vector1::IndexType;

   const auto* data1 = v1.getData();
   const auto* data2 = v2.getData();
   auto fetch = [=] __cuda_callable__ ( IndexType i ) { return data1[ i ] * data2[ i ]; };
   return Algorithms::reduce< DeviceType >( ( IndexType ) 0, v1.getSize(),  fetch, std::plus<>{}, ( ResultType ) 0 );
}

} // namespace Benchmarks
} // namespace TNL
