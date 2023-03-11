/////////////////////////////////////////////////////////////////////
//          Copyright Yibo Zhu 2017
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
/////////////////////////////////////////////////////////////////////
#pragma once
// NOLINTBEGIN

#if ((defined(__clang__) || defined(__GNUC__)) && __cplusplus < 201103L) ||    \
    (defined(_MSC_VER) && _MSC_VER < 1800)
#error This library needs at least a C++11 compliant compiler
#endif
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
template <typename T> static constexpr T getBitmask(unsigned int const bits) {
  return static_cast<T>(-(bits != 0)) &
         (static_cast<T>(-1) >> ((sizeof(T) * CHAR_BIT) - bits));
}

#if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1910)
// c++14 impl
static constexpr unsigned int getSetBitsCount(std::uint64_t n) {
  unsigned int count{0};
  while (n) {
    n &= (n - 1);
    count++;
  }
  return count;
}

static constexpr unsigned int getShiftBitsCount(std::uint64_t n) {
  // requires c++14
  unsigned int count{0};
  if (n == 0)
    return count;
  while ((n & 0x1) == 0) {
    n >>= 1;
    ++count;
  }
  return count;
}

#elif __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)
// c++11 impl
static constexpr unsigned int getSetBitsCount(std::uint64_t n) {
  return n == 0 ? 0 : 1 + getSetBitsCount(n & (n - 1));
}

static constexpr unsigned int getShiftBitsCount(std::uint64_t n) {
  return n == 0 ? 0 : ((n & 0x1) == 0 ? 1 + getShiftBitsCount(n >> 1) : 0);
}

#if (__cplusplus == 201103L) && (defined(__clang__) || defined(__GNUC__))
namespace std { // for c+11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace std
#endif

#else
#error This library needs at least a C++11 compliant compiler
#endif

// NOLINTEND
