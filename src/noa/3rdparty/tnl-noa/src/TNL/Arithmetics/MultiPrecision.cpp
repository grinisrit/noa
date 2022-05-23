// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Daniel Simon, dansimon93@gmail.com
 */

#ifdef HAVE_GMP

   #include "MultiPrecision.h"

namespace noa::TNL {
namespace Arithmetics {

/* CONSTRUCTORS */

MultiPrecision::MultiPrecision()
{
   mpf_init( number );
}

MultiPrecision::MultiPrecision( int i )
{
   signed long int sli = i;
   mpf_init_set_si( number, sli );
}

MultiPrecision::MultiPrecision( double d )
{
   mpf_init_set_d( number, d );
}

/* OPERATORS IMPLEMENTATION */

MultiPrecision&
MultiPrecision::operator=( const MultiPrecision& mp )
{
   mpf_set( number, mp.number );
   return *this;
}

MultiPrecision&
MultiPrecision::operator-()
{
   mpf_neg( this->number, this->number );
   return *this;
}

MultiPrecision&
MultiPrecision::operator+=( const MultiPrecision& mp )
{
   mpf_add( this->number, this->number, mp.number );
   return *this;
}

MultiPrecision&
MultiPrecision::operator-=( const MultiPrecision& mp )
{
   mpf_sub( this->number, this->number, mp.number );
   return *this;
}

MultiPrecision&
MultiPrecision::operator*=( const MultiPrecision& mp )
{
   mpf_mul( this->number, this->number, mp.number );
   return *this;
}

MultiPrecision&
MultiPrecision::operator/=( const MultiPrecision& mp )
{
   mpf_div( this->number, this->number, mp.number );
   return *this;
}

MultiPrecision
MultiPrecision::operator+( const MultiPrecision& mp ) const
{
   MultiPrecision result = MultiPrecision( *this );
   result += mp;
   return result;
}

MultiPrecision
MultiPrecision::operator-( const MultiPrecision& mp ) const
{
   MultiPrecision result( *this );
   result -= mp;
   return result;
}

MultiPrecision
MultiPrecision::operator*( const MultiPrecision& mp ) const
{
   MultiPrecision result( *this );
   result *= mp;
   return result;
}

MultiPrecision
MultiPrecision::operator/( const MultiPrecision& mp ) const
{
   MultiPrecision result( *this );
   result /= mp;
   return result;
}

bool
MultiPrecision::operator==( const MultiPrecision& mp ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, mp.number ) == 0 )
      return true;
   else
      return false;
}

bool
MultiPrecision::operator!=( const MultiPrecision& mp ) const
{
   return ! ( *this == mp );
}

bool
MultiPrecision::operator<( const MultiPrecision& mp ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, mp.number ) < 0 )
      return true;
   else
      return false;
}

bool
MultiPrecision::operator>( const MultiPrecision& mp ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, mp.number ) > 0 )
      return true;
   else
      return false;
}

bool
MultiPrecision::operator>=( const MultiPrecision& mp ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, mp.number ) >= 0 )
      return true;
   else
      return false;
}

bool
MultiPrecision::operator<=( const MultiPrecision& mp ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, mp.number ) <= 0 )
      return true;
   else
      return false;
}

/* OPERATORS FOR GOOGLE TEST */

bool
MultiPrecision::operator==( const mpf_t& GMPnumber ) const
{
   MultiPrecision m( *this );
   if( mpf_cmp( m.number, GMPnumber ) == 0 )
      return true;
   else
      return false;
}

/* METHODS */

MultiPrecision
MultiPrecision::setPrecision( int precision )
{
   mpf_set_default_prec( precision );
}

void
MultiPrecision::printMP()
{
   int precision = mpf_get_default_prec();
   mpf_out_str( stdout, 10, precision, this->number );
   std::cout << std::endl;
}

/* DESTRUCTOR */

MultiPrecision::~MultiPrecision() {}

}  // namespace Arithmetics
}  // namespace noa::TNL

#endif