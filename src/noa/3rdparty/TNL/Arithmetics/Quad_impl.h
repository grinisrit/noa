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

#include <cmath>
#include <cstdio>

#include "Quad.h"

#define ABS(n) ((n) > 0 ? (n): -(n))

namespace noaTNL {
namespace Arithmetics {

template <class T>
Quad<T>::Quad() {
    zeroQuad(data);
}

template <class T>
Quad<T>::Quad(const T& value) {
    data[0] = value;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
}

template <class T>
Quad<T>::Quad(int value) {
    data[0] = (T)value;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
}

template <class T>
Quad<T>::Quad(const Quad<T>& other) {
    data[0] = other[0];
    data[1] = other[1];
    data[2] = other[2];
    data[3] = other[3];
}

template <class T>
T& Quad<T>::operator [](int idx) {
    return data[idx];
}

template <class T>
const T& Quad<T>::operator [](int idx) const{
    return data[idx];
}

template <class T>
Quad<T>& Quad<T>::operator =(const Quad<T>& rhs) {
    data[0] = rhs[0];
    data[1] = rhs[1];
    data[2] = rhs[2];
    data[3] = rhs[3];
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator +=(const Quad<T>& rhs) {
    quadAddAccurate(data, rhs.data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator -=(const Quad<T>& rhs) {
    quadAddAccurate(data, (-rhs).data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator *=(const Quad<T>& rhs) {
    quadMul(data, rhs.data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator /=(const Quad<T>& rhs) {
    quadDiv(data, rhs.data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator =(const T& rhs) {
    data[0] = rhs;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator +=(const T& rhs) {
    doublePlusQuad(rhs, data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator -=(const T& rhs) {
    doublePlusQuad(-rhs, data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator *=(const T& rhs) {
    doubleTimesQuad(rhs, data, data);
    return *this;
}

template <class T>
Quad<T>& Quad<T>::operator /=(const T& rhs) {
    quadDivDouble(data, rhs, data);
    return *this;
}

template <class T>
Quad<T> Quad<T>::operator +(const Quad<T>& value) const{
    Quad<T> qd(*this);
    qd += value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator -(const Quad<T>& value) const{
    Quad<T> qd(*this);
    qd -= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator *(const Quad<T>& value) const{
    Quad<T> qd(*this);
    qd *= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator /(const Quad<T>& value) const{
    Quad<T> qd(*this);
    qd /= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator +(const T& value) const {
    Quad<T> qd(*this);
    qd += value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator -(const T& value) const {
    Quad<T> qd(*this);
    qd -= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator *(const T& value) const {
    Quad<T> qd(*this);
    qd *= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator /(const T& value) const {
    Quad<T> qd(*this);
    qd /= value;
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator +() {
    Quad<T> qd(*this);
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator -() {
    Quad<T> qd(*this);
    qd[0] = -qd[0];
    qd[1] = -qd[1];
    qd[2] = -qd[2];
    qd[3] = -qd[3];
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator +() const {
    Quad<T> qd(*this);
    return qd;
}

template <class T>
Quad<T> Quad<T>::operator -() const {
    Quad<T> qd(*this);
    qd[0] = -qd[0];
    qd[1] = -qd[1];
    qd[2] = -qd[2];
    qd[3] = -qd[3];
    return qd;
}

template <class T>
bool Quad<T>::operator ==(const Quad<T>& rhs) const {
    if (data[0] == rhs[0] && data[1] == rhs[1] && data[2] == rhs[2] && data[3] == rhs[3]) {
        return true;
    }
    return false;
}

template <class T>
bool Quad<T>::operator !=(const Quad<T>& rhs) const {
    return !(*this == rhs);
}

template <class T>
bool Quad<T>::operator <(const Quad<T>& rhs) const {
    Quad<T> qd(*this);
    qd -= rhs;
    if (qd[0] < 0.) {
        return true;
    }
    return false;
}

template <class T>
bool Quad<T>::operator >(const Quad<T>& rhs) const {
    Quad<T> qd(*this);
    qd -= rhs;
    if (qd[0] > 0.) {
        return true;
    }
    return false;
}

template <class T>
bool Quad<T>::operator >=(const Quad<T>& rhs) const {
    Quad<T> qd(*this);
    qd -= rhs;
    if (qd[0] >= 0.) {
        return true;
    }
    return false;
}

template <class T>
bool Quad<T>::operator <=(const Quad<T>& rhs) const  {
    Quad<T> qd(*this);
    qd -= rhs;
    if (qd[0] <= 0.) {
        return true;
    }
    return false;
}

template <class T>
Quad<T>::operator T() const{
    return data[0];
}

template <typename T>
Quad<T> operator+(const T& v1, const Quad<T>& v2) {
    Quad<T> qd(v1);
    qd += v2;
    return qd;
}

template <typename T>
Quad<T> operator-(const T& v1, const Quad<T>& v2) {
    Quad<T> qd(v1);
    qd -= v2;
    return qd;
}

template <typename T>
Quad<T> operator*(const T& v1, const Quad<T>& v2) {
    Quad<T> qd(v1);
    qd *= v2;
    return qd;
}

template <typename T>
Quad<T> operator/(const T& v1, const Quad<T>& v2) {
    Quad<T> qd(v1);
    qd /= v2;
    return qd;
}

template <typename T>
Quad<T> abs(const Quad<T>& value) {
    Quad<T> qd(value);
    if (value[0] < 0) {
        qd = -qd;
    }
    return qd;
}

template <typename T>
Quad<T> sqrt(const Quad<T>& value) {
    Quad<T> qd(value);
    Quad<T> x(1/sqrt((T)qd));
    Quad<T> step;
    //TODO zjednodušit dělení 2
    step = x * (1. - qd * x * x);
    step[0] /= 2;
    step[1] /= 2;
    step[2] /= 2;
    step[3] /= 2;
    x += step;
    step = x * (1. - qd * x * x);
    step[0] /= 2;
    step[1] /= 2;
    step[2] /= 2;
    step[3] /= 2;
    x += step;
    step = x * (1. - qd * x * x);
    step[0] /= 2;
    step[1] /= 2;
    step[2] /= 2;
    step[3] /= 2;
    x += step;
    qd *= x;
    return qd;
}

template <typename T>
void threeThreeSum(T i0, T i1, T i2, T *o0, T *o1, T *o2) {
    twoSum(i0, i1, &i1, &i0); // 1
    twoSum(i1, i2, o0, &i1); // 2
    twoSum(i0, i1, o1, o2); // 3
}

template <typename T>
void threeTwoSum(T i0, T i1, T i2, T *o0, T *o1) {
    twoSum(i0, i1, &i1, &i0); // 1
    twoSum(i1, i2, o0, &i1); // 2
    *o1 = i1 + i0; // 3
}

template <typename T>
void fourTwoSum(T i0, T i1, T i2, T i3, T *o0, T *o1) {
    twoSum(i0, i2, &i0, &i2); // 1
    twoSum(i1, i3, &i1, &i3); // 2
    i2 += i1; // 3
    quickTwoSum(i0, i2, &i0, &i2); // 4
    i3 += i2; // 5
    quickTwoSum(i0, i3, o0, o1); // 6
}

template <typename T>
void sixThreeSum(T i0, T i1, T i2, T i3, T i4, T i5, T *o0, T *o1,
                 T *o2) {
    threeThreeSum(i0, i1, i2, &i2, &i0, &i1); // 1
    threeThreeSum(i3, i4, i5, &i5, &i3, &i4); // 2
    twoSum(i2, i5, o0, &i5); // 3
    twoSum(i0, i3, &i0, &i3); // 4
    twoSum(i0, i5, o1, &i5); // 5
    *o2 = i1 + i4 + i3 + i5; // 6
}

template <typename T>
void sixTwoSum(T i0, T i1, T i2, T i3, T i4, T i5, T *o0, T *o1) {
    threeTwoSum(i0, i1, i2, &i1, &i0);	// 1
    threeTwoSum(i3, i4, i5, &i4, &i3);	// 2
    twoSum(i1, i4, o0, &i1);	// 3
    *o1 = i0 + i3 + i1;	// 4
}

template <typename T>
void nineTwoSum(T i0, T i1, T i2, T i3, T i4, T i5, T i6, T i7,
                T i8, T *o0, T *o1) {
    twoSum(i5, i6, &i5, &i6); // 1
    twoSum(i4, i7, &i4, &i7); // 2
    twoSum(i1, i2, &i1, &i2); // 3
    twoSum(i0, i3, &i0, &i3); // 4
    fourTwoSum(i4, i7, i5, i6, &i4, &i7); // 5
    fourTwoSum(i0, i3, i1, i2, &i0, &i3); // 6
    fourTwoSum(i0, i3, i4, i7, &i0, &i3); // 7
    threeTwoSum(i3, i0, i8, o0, o1); // 8
}

template <typename T>
void doubleAccumulate(T i0, T i1, T i2, T *o0, T *o1, T *o2) {
    twoSum(i1, i2, o0, o2);
    twoSum(i0, *o0, o0, o1);
    if (*o1 == 0) {
        *o1 = *o0;
        *o0 = 0;
    }
    if (*o2 == 0) {
        *o2 = *o1;
        *o1 = *o0;
        *o0 = 0;
    }
}

template <typename T>
void quickTwoSum(T a, T b, T *s, T *e) {
    *s = a + b;
    *e = b - (*s - a);
}

template <typename T>
void twoSum(T a, T b, T *s, T *e) {
    *s = a + b;
    T v = *s - a;
    *e = (a - (*s - v)) + (b - v);
}

template <typename T>
void split(T a, T *a_hi, T *a_lo) {
    T t = 134217729 * a;
    *a_hi = t - (t - a);
    *a_lo = a - *a_hi;
}

template <typename T>
void twoProd(T a, T b, T *p, T *e) {
    *p = a * b;
    T a_hi, a_lo, b_hi, b_lo;
    split(a, &a_hi, &a_lo);
    split(b, &b_hi, &b_lo);
    *e = ((a_hi * b_hi - *p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

template <typename T>
void renormalize(T *a, T *b) {
    T s;
    T t[5];
    int k = 0;
    T e;
    quickTwoSum(a[3], a[4], &s, t + 4);
    quickTwoSum(a[2], s, &s, t + 3);
    quickTwoSum(a[1], s, &s, t + 2);
    quickTwoSum(a[0], s, t, t + 1);
    s = *t;
    int i;
    zeroQuad(b);
    for (i = 1; i < 5; i++) {
        quickTwoSum(s, t[i], &s, &e);
        if (s != 0) {
            b[k] = s;
            s = e;
            k++;
        }
    }
}

template <typename T>
void doublePlusQuad(T b, const T *a, T *s) {
    T m[5];
    T e = b;
    int i;
    for (i = 0; i < 4; i++) {
        twoSum(e, a[i], m + i, &e);
    }
    m[4] = e;
    renormalize(m, s);
}

template <typename T>
void doubleTimesQuad(T b, const T *a, T *s) {
    T m[7];
    twoProd(b, a[0], m, m + 1); // 1
    twoProd(b, a[1], m + 2, m + 3); // 2
    twoSum(m[1], m[2], m + 1, m + 2); // 3
    twoProd(b, a[2], m + 4, m + 5); // 4
    threeThreeSum(m[4], m[3], m[2], m + 2, m + 3, m + 4); // 5
    m[6] = b * a[3]; // 6
    threeTwoSum(m[6], m[5], m[3], m + 3, m + 5); // 7
    m[4] += m[5]; // 8
    renormalize(m, s);
}

template <typename T>
void quadDivDouble(const T *a, T b, T *s) {
    //double b1[] = {b, 0, 0, 0};
    //quadDiv(a, b1, s);
    T m[13];
    int i; // ten půjde odstranit
    m[5] = a[0];
    m[6] = a[1];
    m[7] = a[2];
    m[8] = a[3];
    m[11] = 0;
    m[12] = 0;
    for (i = 0; i < 5; i++) {
        m[i] = m[5] / b;
        twoProd(-m[i], b, m + 9, m + 10);
        //doubleTimesQuad(-m[i], b, m + 9);
        quadAddAccurate(m + 5, m + 9, m + 5);
    }
    renormalize(m, s);
}

template <typename T>
void quadAdd(const T *a, const T *b, T *s) {
    T m[8];
    twoSum(a[0], b[0], m, m + 1); // 1
    twoSum(a[1], b[1], m + 2, m + 3); // 2
    twoSum(m[2], m[1], m + 1, m + 2); // 3
    twoSum(a[2], b[2], m + 4, m + 5); // 4
    // blok 1							   5
    threeThreeSum(m[4], m[3], m[2], m + 2, m + 3, m + 4);
    twoSum(a[3], b[3], m + 6, m + 7); // 6
    // blok 2							   7
    threeTwoSum(m[6], m[5], m[3], m + 3, m + 5);
    m[4] += m[5] + m[7];		// 8
    renormalize(m, s);
}

template <typename T>
void quadAddAccurate(const T *a, const T *b, T *s) {
    T m[11];
    int i = 0;
    int j = 0;
    int k = 0;
    for (;i < 4 && j < 4;k++) {
        if (ABS(a[i]) > ABS(b[j])) {
            m[k] = a[i];
            i++;
        } else {
            m[k] = b[j];
            j++;
        }
    }
    for (; i < 4; i++) {
        m[k] = a[i];
        k++;
    }
    for (; j < 4; j++) {
        m[k] = b[j];
        k++;
    }
    m[9] = 0.;
    m[10] = 0.;
    k = 0;
    for (i = 0; k < 4 && i < 8; i++) {
        doubleAccumulate(m[9], m[10], m[i], m + 8, m + 9, m + 10);
        if (m[8] != 0) {
            m[k] = m[8];
            k++;
        }
    }
    m[k] = m[9];
    m[k + 1] = m[10];
    for (i = k + 2; i < 5; i++) {
        m[i] = 0;
    }
    renormalize(m, s);
}

template <typename T>
void quadMul(const T *a, const T *b, T *s) {
    T m[20];
    twoProd(a[0], b[0], m, m + 1); // 1
    twoProd(a[0], b[1], m + 2, m + 3); // 2
    twoProd(a[1], b[0], m + 4, m + 5); // 3
    threeThreeSum(m[1], m[2], m[4], m + 1, m + 2, m + 4); // 4
    twoProd(a[0], b[2], m + 6, m + 7); // 5
    twoProd(a[1], b[1], m + 8, m + 9); // 6
    twoProd(a[2], b[0], m + 10, m + 11); // 7
    sixThreeSum(m[2], m[3], m[5], m[6], m[8], m[10], m + 2, m + 3, m + 5); // 8
    twoProd(a[0], b[3], m + 12, m + 13); // 9
    twoProd(a[1], b[2], m + 14, m + 15); // 10
    twoProd(a[2], b[1], m + 16, m + 17); // 11
    twoProd(a[3], b[0], m + 18, m + 19); // 12
    nineTwoSum(m[4], m[3], m[7], m[9], m[11], m[12], m[14], m[16], m[18], m + 3, m + 4); // 13
    m[4] += m[5] + m[13] + m[15] + m[17] + m[19] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1]; // 14
    renormalize(m, s);
}

template <typename T>
void quadMulQuick(const T *a, const T *b, T *s) {
    T m[12];
    twoProd(a[0], b[0], m, m + 1); // 1
    twoProd(a[0], b[1], m + 2, m + 3); // 2
    twoProd(a[1], b[0], m + 4, m + 5); // 3
    threeThreeSum(m[1], m[2], m[4], m + 1, m + 2, m + 4); // 4
    twoProd(a[0], b[2], m + 6, m + 7); // 5
    twoProd(a[1], b[1], m + 8, m + 9); // 6
    twoProd(a[2], b[0], m + 10, m + 11); // 7
    sixTwoSum(m[2], m[3], m[5], m[6], m[8], m[10], m + 2, m + 3); // 8
    m[3] += m[4] + m[7] + m[9] + m[11] + a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];  // 9
    m[4] = 0;	// 10
    renormalize(m, s);
}

template <typename T>
void quadDiv(const T *a, const T *b, T *s) {
    T m[13];
    //double n[4];
    //double k[4];
    int i; // ten půjde odstranit
    m[5] = a[0];
    m[6] = a[1];
    m[7] = a[2];
    m[8] = a[3];
    for (i = 0; i < 5; i++) {
        m[i] = m[5] / b[0];
        doubleTimesQuad(-m[i], b, m + 9);
        quadAddAccurate(m + 5, m + 9, m + 5);
    }
    renormalize(m, s);
}

template <typename T>
void zeroQuad(T *a) {
    a[0] = 0;
    a[1] = 0;
    a[2] = 0;
    a[3] = 0;
}

template <typename T>
void printQuad(T *a) {
    printf("%.15le + %.15le + %.15le + %.15le\n", a[0], a[1], a[2], a[3]);
}

} // namespace Arithmetics
} // namespace noaTNL
