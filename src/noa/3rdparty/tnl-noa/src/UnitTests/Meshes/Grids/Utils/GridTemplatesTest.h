#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/GridDetails/Templates/Functions.h>
#include "../CoordinateIterator.h"

void testCombination(const int k, const int n, const int expectation) {
   EXPECT_EQ(TNL::Meshes::Templates::combination(k, n), expectation) << k << " " << n;
}

TEST(TemplatesTestSuite, CombinationsTest) {
  testCombination(0, 1, 1);
  testCombination(1, 1, 1);

  testCombination(0, 2, 1);
  testCombination(1, 2, 2);
  testCombination(2, 2, 1);

  testCombination(0, 3, 1);
  testCombination(1, 3, 3);
  testCombination(2, 3, 3);
  testCombination(3, 3, 1);

  testCombination(0, 4, 1);
  testCombination(1, 4, 4);
  testCombination(2, 4, 6);
  testCombination(3, 4, 4);
  testCombination(4, 4, 1);
}

void testCombinationSum(const int k, const int n, const int expectation) {
   EXPECT_EQ(TNL::Meshes::Templates::firstKCombinationSum(k, n), expectation) << k << " " << n;
}

TEST(TemplatesTestSuite, FirstKCombinationsTest) {
  testCombinationSum(0, 1, 0);
  testCombinationSum(1, 1, 1);

  testCombinationSum(0, 2, 0);
  testCombinationSum(1, 2, 1);
  testCombinationSum(2, 2, 3);

  testCombinationSum(0, 3, 0);
  testCombinationSum(1, 3, 1);
  testCombinationSum(2, 3, 4);
  testCombinationSum(3, 3, 7);

  testCombinationSum(0, 4, 0);
  testCombinationSum(1, 4, 1);
  testCombinationSum(2, 4, 5);
  testCombinationSum(3, 4, 11);
  testCombinationSum(4, 4, 15);
}

template<int Size>
using Coordinate = TNL::Containers::StaticVector<Size, int>;

template<int Size>
void testIndexCollapse(const int base) {
   SCOPED_TRACE("Coordinate size: " + TNL::convertToString(Size));
   SCOPED_TRACE("Base: " + TNL::convertToString(base));

   const int halfBase = base >> 1;
   Coordinate<Size> start, end;

   for (int i = 0; i < Size; i++) {
      start[i] = -halfBase;
      // Want to traverse
      end[i] = halfBase + 1;
   }

   CoordinateIterator<int, Size> iterator(start, end);

   int index = 0;

   do {
      EXPECT_EQ(TNL::Meshes::Templates::makeCollapsedIndex(base, iterator.getCoordinate()), index)
         << base << " " << index << " " << iterator.getCoordinate();
      index++;
   } while (!iterator.next());
}

TEST(TemplatesTestSuite, IndexCollapseTest) {
   testIndexCollapse<1>(3);
   testIndexCollapse<2>(3);
   testIndexCollapse<3>(3);
   testIndexCollapse<4>(3);

   testIndexCollapse<1>(5);
   testIndexCollapse<2>(5);
   testIndexCollapse<3>(5);
   testIndexCollapse<4>(5);
}

void testPower(const size_t value, const size_t power, const size_t expectation) {
   EXPECT_EQ(TNL::Meshes::Templates::pow(value, power), expectation) << value<< " " << power;
}

TEST(TemplatesTestSuite, PowerTest) {
  testPower(0, 1, 0);
  testPower(1, 1, 1);

  testPower(0, 2, 0);
  testPower(1, 2, 1);
  testPower(2, 2, 4);

  testPower(0, 3, 0);
  testPower(1, 3, 1);
  testPower(2, 3, 8);
  testPower(3, 3, 27);

  testPower(0, 4, 0);
  testPower(1, 4, 1);
  testPower(2, 4, 16);
  testPower(3, 4, 81);
  testPower(4, 4, 256);
}

#endif
