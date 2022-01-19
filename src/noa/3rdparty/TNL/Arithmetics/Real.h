// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <cmath>
#include <TNL/Experimental/Arithmetics/FlopsCounter.h>

namespace TNL {

template< class T > class Real
{
   T data;

   public:

   Real()
   : data( 0 )
     {};

   template< class S >Real( const S& d )
   : data( d )
     {}

   Real( const Real& v )
   : data( v. data )
     {};

   T& Data()
   {
      return data;
   };

   const T& Data() const
   {
      return data;
   };

   const Real& operator += ( const Real& v )
   {
      data += v. data;
      tnl_flops_counter. recordAdding();
      return *this;
   };

	template< class S > const Real& operator += ( const S& v )
	{
		data += v;
		tnl_flops_counter. recordAdding();
		return *this;
	}

	const Real& operator -= ( const Real& v )
	{
		data -= v. data;
		tnl_flops_counter. recordAdding();
		return *this;
	};

	template< class S > const Real& operator -= ( const S& v )
	{
		data -= v;
		tnl_flops_counter. recordAdding();
		return *this;
	}

	const Real& operator *= ( const Real& v )
	{
		data *= v. data;
		tnl_flops_counter. recordMultiplying();
		return *this;
	};

	template< class S > const Real& operator *= ( const S& v )
	{
		data *= v;
		tnl_flops_counter. recordMultiplying();
		return *this;
	}

	const Real& operator /= ( const Real& v )
	{
		data /= v. data;
		tnl_flops_counter. recordDividing();
		return *this;
	};

	template< class S > const Real& operator /= ( const S& v )
	{
		data /= v;
		tnl_flops_counter. recordDividing();
		return *this;
	}

	const Real& operator + ( const Real& v ) const
	{
		return Real( *this ) += v;
	};

	template< class S > const Real& operator + ( const S& v ) const
	{
		return Real( *this ) += v;
	}

	const Real& operator - ( const Real& v ) const
	{
		return Real( *this ) -= v;
	};

	template< class S > const Real& operator - ( const S& v ) const
	{
		return Real( *this ) -= v;
	}

	const Real& operator - () const
	{
		return Real( *this ) *= -1.0;
	};

	const Real& operator * ( const Real& v ) const
	{
		return Real( *this ) *= v;
	};

	template< class S > const Real& operator * ( const S& v ) const
	{
		return Real( *this ) *= v;
	}

	const Real& operator / ( const Real& v ) const
	{
		return Real( *this ) /= v;
	};

	template< class S > const Real& operator / ( const S& v ) const
	{
		return Real( *this ) /= v;
	}

	const Real& operator = ( const Real& v )
	{
		data = v. data;
		return *this;
	};

	const Real& operator = ( const T& v )
	{
		data = v;
		return *this;
	};

	bool operator == ( const Real& v ) const
	{
		return data == v.data;
	};

	bool operator == ( const T& v ) const
	{
		return data == v;
	};

	bool operator != ( const T& v ) const
	{
		return data != v;
	};

	bool operator != ( const Real& v ) const
	{
		return data != v.data;
	};

	bool operator <= ( const T& v ) const
	{
		return data <= v;
	};

	bool operator <= ( const Real& v ) const
	{
		return data <= v.data;
	};

	bool operator >= ( const T& v ) const
	{
		return data >= v;
	};

	bool operator >= ( const Real& v ) const
	{
		return data >= v.data;
	};

	bool operator < ( const T& v ) const
	{
		return data < v;
	};

	bool operator < ( const Real& v ) const
	{
		return data < v.data;
	};

	bool operator > ( const T& v ) const
	{
		return data > v;
	};

	bool operator > ( const Real& v ) const
	{
		return data > v.data;
	};

	bool operator || ( const Real& v ) const
    {
		return data || v.data;
    };

	bool operator && ( const Real& v ) const
    {
		return data && v.data;
    };

	bool operator ! () const
   {
	   return ! data;
   };

    /*operator bool () const
    {
	   return ( bool ) data;
	};*/

	operator int () const
	{
	   return ( int ) data;
	};

	/*operator float () const
	{
		return ( float ) data;
	};*/

	operator double () const
    {
		return ( double ) data;
    };

};

template< class T, class S > const Real< T >& operator + ( const S& v1, const Real< T >& v2 )
{
   return Real< T >( v1 ) += v2. Data();
};

template< class T, class S > const Real< T >& operator - ( const S& v1, const Real< T >& v2 )
{
   return Real< T >( v1 ) -= v2. Data();
};

template< class T, class S > const Real< T >& operator * ( const S& v1, const Real< T >& v2 )
{
   return Real< T >( v1 ) *= v2. Data();
};

template< class T, class S > const Real< T >& operator / ( const S& v1, const Real< T >& v2 )
{
   return Real< T >( v1 ) /= v2. Data();
};

template< class T > bool operator == ( const T& v1, const Real< T >& v2 )
{
   return v1 == v2. Data();
};

template< class T > bool operator != ( const T& v1, const Real< T >& v2 )
{
   return v1 != v2. Data();
};

template< class T > bool operator <= ( const T& v1, const Real< T >& v2 )
{
   return v1 <= v2. Data();
};

template< class T > bool operator >= ( const T& v1, const Real< T >& v2 )
{
   return v1 >= v2. Data();
};

template< class T > bool operator < ( const T& v1, const Real< T >& v2 )
{
   return v1 < v2. Data();
};

template< class T > bool operator > ( const T& v1, const Real< T >& v2 )
{
   return v1 > v2. Data();
};

template< class T > const Real< T > fabs( const Real< T >& v )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( fabs( v. Data() ) );
};

template< class T > const Real< T > sqrt( const Real< T >& v )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::sqrt( v. Data() ) );
};

template< class T > const Real< T > pow( const Real< T >& x, const Real< T >& exp )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::pow( x. Data(), exp. Data() ) );
};

template< class T > const Real< T > pow( const Real< T >& x, const T& exp )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::pow( x. Data(), exp ) );
};

template< class T > const Real< T > cos( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::cos( x. Data() ) );
};

template< class T > const Real< T > sin( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::sin( x. Data() ) );
};

template< class T > const Real< T > tan( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::tan( x. Data() ) );
};

template< class T > const Real< T > acos( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::acos( x. Data() ) );
};

template< class T > const Real< T > asin( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::asin( x. Data() ) );
};

template< class T > const Real< T > atan( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::atan( x. Data() ) );
};

template< class T > const Real< T > atan2( const Real< T >& x, const Real< T >& exp )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::atan2( x. Data(), exp. Data() ) );
};

template< class T > const Real< T > atan2( const Real< T >& x, const T& exp )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::atan2( x. Data(), exp ) );
};


template< class T > const Real< T > cosh( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::cosh( x. Data() ) );
};

template< class T > const Real< T > sinh( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::sinh( x. Data() ) );
};

template< class T > const Real< T > tanh( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::tanh( x. Data() ) );
};


template< class T > const Real< T > exp( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::exp( x. Data() ) );
};

template< class T > const Real< T > log( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::log( x. Data() ) );
};

template< class T > const Real< T > log10( const Real< T >& x )
{
   tnl_flops_counter. recordFunction();
   return Real< T >( std::log10( x. Data() ) );
};

template< class T >
std::ostream& operator << ( std::ostream& str, const Real< T >& v )
{
   str << v. Data();
   return str;
};

typedef Real< float > tnlFloat;
typedef Real< double > tnlDouble;

} // namespace TNL
