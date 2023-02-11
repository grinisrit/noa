
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include "../CoordinateIterator.h"

template<typename Iterator>
void test(Iterator& iterator, const typename Iterator::Coordinate& coordinate, bool canIterate, bool next) {
  SCOPED_TRACE("Coordinate: " + TNL::convertToString(coordinate));
  SCOPED_TRACE("Can iterate: " + TNL::convertToString(canIterate));
  SCOPED_TRACE("Next: " + TNL::convertToString(next));

  EXPECT_EQ(iterator.getCoordinate(), coordinate);
  EXPECT_EQ(iterator.canIterate(), canIterate);
  EXPECT_EQ(iterator.next(), next);
}

TEST(CoordinateIteratorTest, ZeroStartIterator1D) {
  CoordinateIterator<int, 1> iterator({ 0 }, { 3 });

  test(iterator, { 0 }, true, false);
  test(iterator, { 1 }, true, false);
  test(iterator, { 2 }, true, true);
}

TEST(CoordinateIteratorTest, ZeroStartIterator2D) {
  CoordinateIterator<int, 2> iterator({ 0, 0 }, { 3, 3 });

  test(iterator, { 0, 0 }, true, false);
  test(iterator, { 1, 0 }, true, false);
  test(iterator, { 2, 0 }, true, false);
  test(iterator, { 0, 1 }, true, false);
  test(iterator, { 1, 1 }, true, false);
  test(iterator, { 2, 1 }, true, false);
  test(iterator, { 0, 2 }, true, false);
  test(iterator, { 1, 2 }, true, false);
  test(iterator, { 2, 2 }, true, true);
}

TEST(CoordinateIteratorTest, ZeroStartIterator3D) {
  CoordinateIterator<int, 3> iterator({ 0, 0, 0 }, { 2, 2, 2 });

  test(iterator, { 0, 0, 0 }, true, false);
  test(iterator, { 1, 0, 0 }, true, false);
  test(iterator, { 0, 1, 0 }, true, false);
  test(iterator, { 1, 1, 0 }, true, false);
  test(iterator, { 0, 0, 1 }, true, false);
  test(iterator, { 1, 0, 1 }, true, false);
  test(iterator, { 0, 1, 1 }, true, false);
  test(iterator, { 1, 1, 1 }, true, true);
}

TEST(CoordinateIteratorTest, SomeStartIterator1D) {
  CoordinateIterator<int, 1> iterator({ 1 }, { 3 });

  test(iterator, { 1 }, true, false);
  test(iterator, { 2 }, true, true);
}

TEST(CoordinateIteratorTest, SomeStartIterator2D) {
  CoordinateIterator<int, 2> iterator({ 1, 1 }, { 3, 3 });

  test(iterator, { 1, 1 }, true, false);
  test(iterator, { 2, 1 }, true, false);
  test(iterator, { 1, 2 }, true, false);
  test(iterator, { 2, 2 }, true, true);
}

TEST(CoordinateIteratorTest, SomeStartIterator3D) {
  CoordinateIterator<int, 3> iterator({ 1, 1, 1 }, { 3, 3, 3 });

  test(iterator, { 1, 1, 1 }, true, false);
  test(iterator, { 2, 1, 1 }, true, false);
  test(iterator, { 1, 2, 1 }, true, false);
  test(iterator, { 2, 2, 1 }, true, false);
  test(iterator, { 1, 1, 2 }, true, false);
  test(iterator, { 2, 1, 2 }, true, false);
  test(iterator, { 1, 2, 2 }, true, false);
  test(iterator, { 2, 2, 2 }, true, true);
}

#endif