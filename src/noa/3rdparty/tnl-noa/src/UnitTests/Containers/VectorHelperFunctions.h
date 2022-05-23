#pragma once

#include <TNL/Math.h>
#include <TNL/TypeTraits.h>
#include <TNL/Devices/Host.h>

template< typename Vector >
void setLinearSequence( Vector& deviceVector )
{
#ifdef STATIC_VECTOR
   Vector a;
#else
   using HostVector = typename Vector::template Self< typename Vector::ValueType, TNL::Devices::Host >;
   HostVector a;
   a.setLike( deviceVector );
#endif
#ifdef DISTRIBUTED_VECTOR
   for( int i = 0; i < a.getLocalRange().getSize(); i++ ) {
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = gi;
   }
   for( int i = a.getLocalRange().getSize(); i < a.getLocalView().getSize(); i++ )
      a.getLocalView()[ i ] = -1;  // dummy ghost value
#else
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
#endif
   deviceVector = a;
}

template< typename Vector, typename Value >
void setConstantSequence( Vector& deviceVector, Value v )
{
   deviceVector.setValue( typename Vector::ValueType( v ) );
}

template< typename Vector >
void setOscilatingLinearSequence( Vector& deviceVector )
{
   using HostVector = typename Vector::template Self< typename Vector::ValueType, TNL::Devices::Host >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i % 30 - 15;
   deviceVector = a;
}

template< typename Vector >
void setOscilatingConstantSequence( Vector& deviceVector )
{
   using HostVector = typename Vector::template Self< typename Vector::ValueType, TNL::Devices::Host >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = TNL::sign( i % 30 - 15 );
   deviceVector = a;
}

template< typename Vector >
void setNegativeLinearSequence( Vector& deviceVector )
{
   using HostVector = typename Vector::template Self< typename Vector::ValueType, TNL::Devices::Host >;
   HostVector a;
   a.setLike( deviceVector );
#ifdef DISTRIBUTED_VECTOR
   for( int i = 0; i < a.getLocalRange().getSize(); i++ ) {
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = -gi;
   }
   for( int i = a.getLocalRange().getSize(); i < a.getLocalView().getSize(); i++ )
      a.getLocalView()[ i ] = 1;  // dummy ghost value
#else
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
#endif
   deviceVector = a;
}

template< typename Vector, typename Value >
void setOscilatingSequence( Vector& deviceVector, Value v )
{
#ifdef STATIC_VECTOR
   Vector a;
#else
   using HostVector = typename Vector::template Self< typename Vector::ValueType, TNL::Devices::Host >;
   HostVector a;
   a.setLike( deviceVector );
#endif
#ifdef DISTRIBUTED_VECTOR
   for( int i = 0; i < a.getLocalRange().getSize(); i++ ) {
      const auto gi = a.getLocalRange().getGlobalIndex( i );
      a[ gi ] = v * std::pow( -1, gi );
   }
   for( int i = a.getLocalRange().getSize(); i < a.getLocalView().getSize(); i++ )
      a.getLocalView()[ i ] = 42;  // dummy ghost value
#else
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = v * std::pow( -1, i );
#endif
   deviceVector = a;
}


// specialization for V1 = view
template< typename V1, typename V2,
          std::enable_if_t< TNL::IsViewType< V1 >::value, bool > = true >
void bindOrAssign( V1& v1, V2& v2 )
{
   v1.bind( v2.getView() );
}

// specialization for V1 = vector
template< typename V1, typename V2,
          std::enable_if_t< ! TNL::IsViewType< V1 >::value, bool > = true >
void bindOrAssign( V1& v1, V2& v2 )
{
   v1 = v2;
}


#ifdef HAVE_GTEST
#include "gtest/gtest.h"

template< typename T1, typename T2,
          std::enable_if_t< ! TNL::HasSubscriptOperator< T1 >::value &&
                            ! TNL::HasSubscriptOperator< T2 >::value, bool > = true >
void expect_near( const T1& arg, const T2& expected, double epsilon )
{
   EXPECT_NEAR( arg, expected, epsilon );
}

template< typename T1, typename T2,
          std::enable_if_t< TNL::HasSubscriptOperator< T1 >::value &&
                            ! TNL::HasSubscriptOperator< T2 >::value, bool > = true >
void expect_near( const T1& arg, const T2& expected, double epsilon )
{
   for( int i = 0; i < arg.getSize(); i++ )
      expect_near( arg[ i ], expected, epsilon );
}

template< typename T1, typename T2,
          std::enable_if_t< TNL::HasSubscriptOperator< T1 >::value &&
                            TNL::HasSubscriptOperator< T2 >::value, bool > = true >
void expect_near( const T1& arg, const T2& expected, double epsilon )
{
   ASSERT_EQ( arg.getSize(), expected.getSize() );
   for( int i = 0; i < arg.getSize(); i++ )
      expect_near( arg[ i ], expected[ i ], epsilon );
}
#endif
