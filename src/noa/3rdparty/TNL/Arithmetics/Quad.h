// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Matěj Novotný
 * Daniel Simon, dansimon93@gmail.com
 */

#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Arithmetics {    

template <class T>
class Quad
{
public:
    /*INIT*/
    T data[4];

    Quad();
    explicit Quad(const T&);
    explicit Quad(int);
    Quad(const Quad<T>&);

    /*OVERLOADED OPERATORS*/
    T& operator[](int);
    const T& operator[](int) const;
    Quad<T>& operator =(const Quad<T>&);
    Quad<T>& operator +=(const Quad<T>&);
    Quad<T>& operator -=(const Quad<T>&);
    Quad<T>& operator *=(const Quad<T>&);
    Quad<T>& operator /=(const Quad<T>&);
    Quad<T>& operator =(const T&);
    Quad<T>& operator +=(const T&);
    Quad<T>& operator -=(const T&);
    Quad<T>& operator *=(const T&);
    Quad<T>& operator /=(const T&);
    Quad<T> operator +(const Quad<T>&) const;
    Quad<T> operator -(const Quad<T>&) const;
    Quad<T> operator *(const Quad<T>&) const;
    Quad<T> operator /(const Quad<T>&) const;
    Quad<T> operator +(const T&) const;
    Quad<T> operator -(const T&) const;
    Quad<T> operator *(const T&) const;
    Quad<T> operator /(const T&) const;
    Quad<T> operator +();
    Quad<T> operator -();
    Quad<T> operator +() const;
    Quad<T> operator -() const;
    bool operator ==(const Quad<T>&) const;
    bool operator !=(const Quad<T>&) const;
    bool operator <(const Quad<T>&) const;
    bool operator >(const Quad<T>&) const;
    bool operator >=(const Quad<T>&) const;
    bool operator <=(const Quad<T>&) const;
    explicit operator T() const;
};


template <typename T>
Quad<T> operator +(const T&, const Quad<T>&);
template <typename T>
Quad<T> operator -(const T&, const Quad<T>&);
template <typename T>
Quad<T> operator *(const T&, const Quad<T>&);
template <typename T>
Quad<T> operator /(const T&, const Quad<T>&);

template <typename T>
Quad<T> abs(const Quad<T>&);
template <typename T>
Quad<T> sqrt(const Quad<T>&);

template <typename T>
void quickTwoSum(T a, T b, T *s, T *e); // Addition of two doubles
template <typename T>
void twoSum(T a, T b, T *s, T *e); // Addition of two doubles
template <typename T>
void split(T a, T *a_hi, T *a_lo); // Split double into two 26 bits parts
template <typename T>
void twoProd(T a, T b, T *p, T *e); // Multiplication of two doubles
template <typename T>
void renormalize(T *a, T *b); // Normalization of number a
template <typename T>
void doublePlusQuad(T b, const T *a, T *s); // Addition of double and quad-double
template <typename T>
void doubleTimesQuad(T b, const T *a, T *s); // Multiplication of double and quad-double
template <typename T>
void quadDivDouble(const T *a, T b, T *s); // Division of two doubles
template <typename T>
void quadAdd(const T *a, const T *b, T *s); // Addition of two quad-doubles
template <typename T>
void quadAddAccurate(const T *a, const T *b, T *s); // Addition of two quad-doubles ! slower algorhitm
template <typename T>
void quadMul(const T *a, const T *b, T *s); // Multiplication of two quad-doubles
template <typename T>
void quadMulQuick(const T *a, const T *b, T *s); // Multiplication of two quad-doubles ! faster algorithm
template <typename T>
void quadDiv(const T *a, const T *b, T *s); // Division of two quad-doubles
template <typename T>
void zeroQuad(T *a); // Reset quad-double
template <typename T>
void printQuad(T *a); // Print of quad-double

} // namespace Arithmetics
} //namespace TNL
