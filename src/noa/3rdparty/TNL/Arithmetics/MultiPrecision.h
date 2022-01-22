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

/****
 * Wrapper class for GMP library: https://gmplib.org/
 */

#pragma once

#ifdef HAVE_GMP
#include <gmp.h>
#endif

namespace noaTNL {
namespace Arithmetics {
    
class MultiPrecision
{
   public:

#ifdef HAVE_GMP    
    /* CONSTRUCTORS */
    MultiPrecision(); // initialize number to 0
    explicit MultiPrecision(int); // assignment of signed long integer
    explicit MultiPrecision(double d); // assignment of double

    /* OPERATORS */
    MultiPrecision& operator=(const MultiPrecision& mp);
    MultiPrecision& operator-();
    MultiPrecision& operator+=(const MultiPrecision& mp);
    MultiPrecision& operator-=(const MultiPrecision& mp);
    MultiPrecision& operator*=(const MultiPrecision& mp);
    MultiPrecision& operator/=(const MultiPrecision& mp);
    MultiPrecision operator+(const MultiPrecision& mp) const;
    MultiPrecision operator-(const MultiPrecision& mp) const;
    MultiPrecision operator*(const MultiPrecision& mp) const;
    MultiPrecision operator/(const MultiPrecision& mp) const;
    bool operator==(const MultiPrecision &mp) const;
    bool operator!=(const MultiPrecision &mp) const;
    bool operator<(const MultiPrecision &mp) const;
    bool operator>(const MultiPrecision &mp) const;
    bool operator>=(const MultiPrecision &mp) const;
    bool operator<=(const MultiPrecision &mp) const;
    MultiPrecision& operator++();
    MultiPrecision& operator--();
    MultiPrecision operator++(int);
    MultiPrecision operator--(int);

    /* OPERATORS FOR GOOGLE TEST*/
    bool operator==(const mpf_t &GMPnumber) const;
    
    /* METHODS */
    void printMP();
    static MultiPrecision setPrecision(int); // sets the default precision
    /// TODO void printNumber(int digits, ostream& str = std::cout );

    /* DESTRUCTOR */
    ~MultiPrecision();
    
    mpf_t number;
#endif
};

MultiPrecision abs(const MultiPrecision);
MultiPrecision sqrt(const MultiPrecision);
MultiPrecision cqrt(const MultiPrecision);

} // namespace Arithmetics
} // namespace noaTNL
