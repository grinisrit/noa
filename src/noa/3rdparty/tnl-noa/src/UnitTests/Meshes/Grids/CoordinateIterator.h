
#pragma once

#include <TNL/Containers/StaticVector.h>

template <typename Index, int Size>
class CoordinateIterator {
  public:
   using Coordinate = TNL::Containers::StaticVector<Size, Index>;

   CoordinateIterator(const Coordinate& start, const Coordinate& end): start(start), current(start), end(end) {}

   Coordinate getCoordinate() const { return current; }

   bool next() {
      current[0] += 1;

      Index carry = 0;

      bool isEnded = false;

      for (Index i = 0; i < current.getSize(); i++) {
         current[i] += carry;

         if (current[i] == end[i]) {
            carry = 1;
            current[i] = start[i];

            isEnded = i == current.getSize() - 1;
            continue;
         }

         break;
      }

      return isEnded;
   }

   bool canIterate() {
      for (Index i = 0; i < current.getSize(); i++)
         if (current[i] >= end[i]) return false;

      return true;
   }
  protected:
   Coordinate start, current, end;
};
