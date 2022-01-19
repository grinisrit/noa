// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

enum class Type {
   /* LONG = 0!!! Non zero value rewrites index[1] */
   LONG = 0,
   STREAM = 1,
   VECTOR = 2
};

//#define CSR_ADAPTIVE_UNION

#ifdef CSR_ADAPTIVE_UNION
template< typename Index >
union CSRAdaptiveKernelBlockDescriptor
{
   CSRAdaptiveKernelBlockDescriptor(Index row, Type type = Type::VECTOR, Index index = 0, uint8_t warpsCount = 0) noexcept
   {
      this->index[0] = row;
      this->index[1] = index;
      this->byte[sizeof(Index) == 4 ? 7 : 15] = (uint8_t)type;
   }

   CSRAdaptiveKernelBlockDescriptor(Index row, Type type, Index nextRow, Index maxID, Index minID) noexcept
   {
      this->index[0] = row;
      this->index[1] = 0;
      this->twobytes[sizeof(Index) == 4 ? 2 : 4] = maxID - minID;

      if (type == Type::STREAM)
         this->twobytes[sizeof(Index) == 4 ? 3 : 5] = nextRow - row;

      if (type == Type::STREAM)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b1000000;
      else if (type == Type::VECTOR)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b10000000;
   }

   CSRAdaptiveKernelBlockDescriptor() = default;

   __cuda_callable__ Type getType() const
   {
      if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b1000000 )
         return Type::STREAM;
      if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b10000000 )
         return Type::VECTOR;
      return Type::LONG;
   }

   __cuda_callable__ const Index& getFirstSegment() const
   {
      return index[ 0 ];
   }

   /***
    * \brief Returns number of elements covered by the block.
    */
   __cuda_callable__ const Index getSize() const
   {
      return twobytes[ sizeof(Index) == 4 ? 2 : 4 ];
   }

   /***
    * \brief Returns number of segments covered by the block.
    */
   __cuda_callable__ const Index getSegmentsInBlock() const
   {
      return ( twobytes[ sizeof( Index ) == 4 ? 3 : 5 ] & 0x3FFF );
   }

   __cuda_callable__ uint8_t getWarpIdx() const
   {
      return index[ 1 ];
   }

   __cuda_callable__ uint8_t getWarpsCount() const
   {
      return 1;
   }

   void print( std::ostream& str ) const
   {
      Type type = this->getType();
      str << "Type: ";
      switch( type )
      {
         case Type::STREAM:
            str << " Stream ";
            break;
         case Type::VECTOR:
            str << " Vector ";
            break;
         case Type::LONG:
            str << " Long ";
            break;
      }
      str << " first segment: " << getFirstSegment();
      str << " block end: " << getSize();
      str << " index in warp: " << index[ 1 ];
   }
   Index index[2]; // index[0] is row pointer, index[1] is index in warp
   uint8_t byte[sizeof(Index) == 4 ? 8 : 16]; // byte[7/15] is type specificator
   uint16_t twobytes[sizeof(Index) == 4 ? 4 : 8]; //twobytes[2/4] is maxID - minID
                                                //twobytes[3/5] is nextRow - row
};
#else

template< typename Index >
struct CSRAdaptiveKernelBlockDescriptor
{
   CSRAdaptiveKernelBlockDescriptor( Index firstSegmentIdx,
                                     Type type = Type::VECTOR,
                                     uint8_t warpIdx = 0,
                                     uint8_t warpsCount = 0 ) noexcept
   {
      this->firstSegmentIdx = firstSegmentIdx;
      this->type = ( uint8_t ) type;
      this->warpIdx = warpIdx;
      this->warpsCount = warpsCount;
      /*this->index[0] = row;
      this->index[1] = index;
      this->byte[sizeof(Index) == 4 ? 7 : 15] = (uint8_t)type;*/
   }

   CSRAdaptiveKernelBlockDescriptor( Index firstSegmentIdx,
                                     Type type,
                                     Index lastSegmentIdx,
                                     Index end,
                                     Index begin ) noexcept
   {
      this->firstSegmentIdx = firstSegmentIdx;
      this->warpIdx = 0;
      this->blockSize = end - begin;
      this->segmentsInBlock = lastSegmentIdx - firstSegmentIdx;
      this->type = ( uint8_t ) type;

      /*this->index[0] = row;
      this->index[1] = 0;
      this->twobytes[sizeof(Index) == 4 ? 2 : 4] = maxID - minID;

      if (type == Type::STREAM)
         this->twobytes[sizeof(Index) == 4 ? 3 : 5] = nextRow - row;

      if (type == Type::STREAM)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b1000000;
      else if (type == Type::VECTOR)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b10000000;*/
   }

   CSRAdaptiveKernelBlockDescriptor() = default;

   __cuda_callable__ Type getType() const
   {
      return ( Type ) this->type;
      /*if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b1000000 )
         return Type::STREAM;
      if( byte[ sizeof( Index ) == 4 ? 7 : 15 ] & 0b10000000 )
         return Type::VECTOR;
      return Type::LONG;*/
   }

   __cuda_callable__ const Index& getFirstSegment() const
   {
      return this->firstSegmentIdx;
      //return index[ 0 ];
   }

   /***
    * \brief Returns number of elements covered by the block.
    */
   __cuda_callable__ const Index getSize() const
   {
      return this->blockSize;
      //return twobytes[ sizeof(Index) == 4 ? 2 : 4 ];
   }

   /***
    * \brief Returns number of segments covered by the block.
    */
   __cuda_callable__ const Index getSegmentsInBlock() const
   {
      return this->segmentsInBlock;
      //return ( twobytes[ sizeof( Index ) == 4 ? 3 : 5 ] & 0x3FFF );
   }

   __cuda_callable__ uint8_t getWarpIdx() const
   {
      return this->warpIdx;
   }

   __cuda_callable__ uint8_t getWarpsCount() const
   {
      return this->warpsCount;
   }

   void print( std::ostream& str ) const
   {
      str << "Type: ";
      switch( this->getType() )
      {
         case Type::STREAM:
            str << " Stream ";
            break;
         case Type::VECTOR:
            str << " Vector ";
            break;
         case Type::LONG:
            str << " Long ";
            break;
      }
      str << " first segment: " << this->getFirstSegment();
      str << " block end: " << this->getSize();
      str << " index in warp: " << this->getWarpIdx();
   }

   uint8_t type;
   Index firstSegmentIdx, blockSize, segmentsInBlock;
   uint8_t warpIdx, warpsCount;

   //Index index[2]; // index[0] is row pointer, index[1] is index in warp
   //uint8_t byte[sizeof(Index) == 4 ? 8 : 16]; // byte[7/15] is type specificator
   //uint16_t twobytes[sizeof(Index) == 4 ? 4 : 8]; //twobytes[2/4] is maxID - minID
                                                //twobytes[3/5] is nextRow - row
};

#endif

template< typename Index >
std::ostream& operator<< ( std::ostream& str, const CSRAdaptiveKernelBlockDescriptor< Index >& block )
{
   block.print( str );
   return str;
}
         } // namespace detail
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
