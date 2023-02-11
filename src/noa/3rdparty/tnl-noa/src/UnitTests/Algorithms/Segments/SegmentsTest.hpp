#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template< typename Segments >
void test_SetSegmentsSizes_EqualSizes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   EXPECT_EQ( segments.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments2.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getSize(), segments2.getStorageSize() );
   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsSizes );

   EXPECT_EQ( segments3.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments3.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getSize(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segmentsView.getSegmentSize( i ), segmentSize );
}

template< typename Segments >
void test_SetSegmentsSizes_EqualSizes_EllpackOnly()
{
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;

   Segments segments( segmentsCount, segmentSize );

   EXPECT_EQ( segments.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments2.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getSize(), segments2.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsCount, segmentSize );

   EXPECT_EQ( segments3.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments3.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getSize(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segmentsView.getSegmentSize( i ), segmentSize );
}

template< typename Segments >
void test_reduceAllSegments_MaximumInSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );

   auto view = v.getView();
   auto init = [=] __cuda_callable__ ( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool {
      view[ globalIdx ] =  segmentIdx * 5 + localIdx + 1;
      return true;
   };
   segments.forAllElements( init );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType >result( segmentsCount );

   const auto v_view = v.getConstView();
   auto result_view = result.getView();
   auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) -> IndexType {
      return v_view[ globalIdx ];
   };
   auto reduce = [] __cuda_callable__ ( IndexType& a, const IndexType b ) -> IndexType {
      return TNL::max( a, b );
   };
   auto keep = [=] __cuda_callable__ ( const IndexType i, const IndexType a ) mutable {
      result_view[ i ] = a;
   };
   segments.reduceAllSegments( fetch, reduce, keep, std::numeric_limits< IndexType >::min() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( result.getElement( i ), ( i + 1 ) * segmentSize );

   result_view = 0;
   segments.getView().reduceAllSegments( fetch, reduce, keep, std::numeric_limits< IndexType >::min() );
   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( result.getElement( i ), ( i + 1 ) * segmentSize );
}

#endif
