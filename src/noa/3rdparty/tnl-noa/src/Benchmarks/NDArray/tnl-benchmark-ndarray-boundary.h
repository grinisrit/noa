// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Assert.h>
#include <TNL/Math.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Containers/NDArray.h>

#include <TNL/Benchmarks/Benchmarks.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Containers;
using std::index_sequence;

using value_type = float;
//using index_type = std::size_t;
using index_type = unsigned;

template< typename Array >
void expect_eq_chunked( Array& a, Array& b )
{
   // TODO: use something like EXPECT_EQ
   TNL_ASSERT_EQ( a.getSize(), b.getSize(), "array sizes don't match" );
   if( a.getSize() != b.getSize() )
      return;

   using IndexType = typename Array::IndexType;

   const IndexType chunk_size = 4096;
   for( IndexType c = 0; c < (IndexType) roundUpDivision( a.getSize(), chunk_size ); c++ ) {
      const typename Array::IndexType this_chunk_size = TNL::min( chunk_size, a.getSize() - c * chunk_size );
      Array a_chunk( &a[ c * chunk_size ], this_chunk_size );
      Array b_chunk( &b[ c * chunk_size ], this_chunk_size );
      // TODO: use something like EXPECT_EQ
      TNL_ASSERT_EQ( a_chunk, b_chunk, "chunks are not equal" );
   }
}

template< typename Array >
void expect_eq( Array& a, Array& b )
{
   if( std::is_same< typename Array::DeviceType, TNL::Devices::Cuda >::value ) {
      using HostArray = typename Array::template Self< typename Array::ValueType, TNL::Devices::Host >;
      HostArray a_host, b_host;
      a_host = a;
      b_host = b;
      expect_eq_chunked( a_host, b_host );
   }
   else {
      expect_eq_chunked( a, b );
   }
}

template< typename Device >
const char* performer()
{
   if( std::is_same< Device, Devices::Host >::value )
      return "CPU";
   else if( std::is_same< Device, Devices::Cuda >::value )
      return "GPU";
   else
      return "unknown";
}

void reset() {}

// NOTE: having the sizes as function parameters keeps the compiler from treating them
// as "compile-time constants" and thus e.g. optimizing the 1D iterations with memcpy

template< typename Device >
void benchmark_1D( Benchmark<>& benchmark, index_type size = 500000000 )
{
   NDArray< value_type,
            SizesHolder< index_type, 0 >,
            std::make_index_sequence< 1 >,
            Device > a, b;
   a.setSizes( size );
   b.setSizes( size );
   a.getStorageArray().setValue( -1 );
   b.getStorageArray().setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [&]() {
      a.forBoundary( [=] __cuda_callable__ ( index_type i ) mutable { a_view( i ) = b_view( i ); } );
      a.forInternal( [=] __cuda_callable__ ( index_type i ) mutable { a_view( i ) = b_view( i ); } );
   };

   const double datasetSize = 2 * size * sizeof(value_type) / oneGB;
   benchmark.setOperation( "1D", datasetSize );
   benchmark.time< Device >( reset, performer< Device >(), f );

   expect_eq( a.getStorageArray(), b.getStorageArray() );
}

template< typename Device >
void benchmark_2D( Benchmark<>& benchmark, index_type size = 22333 )
{
   NDArray< value_type,
            SizesHolder< index_type, 0, 0 >,
            std::make_index_sequence< 2 >,
            Device > a, b;
   a.setSizes( size, size );
   b.setSizes( size, size );
   a.getStorageArray().setValue( -1 );
   b.getStorageArray().setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [&]() {
      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j ) mutable { a_view( i, j ) = b_view( i, j ); } );
      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j ) mutable { a_view( i, j ) = b_view( i, j ); } );
   };

   const double datasetSize = 2 * std::pow( size, 2 ) * sizeof(value_type) / oneGB;
   benchmark.setOperation( "2D", datasetSize );
   benchmark.time< Device >( reset, performer< Device >(), f );

   expect_eq( a.getStorageArray(), b.getStorageArray() );
}

template< typename Device >
void benchmark_3D( Benchmark<>& benchmark, index_type size = 800 )
{
   NDArray< value_type,
            SizesHolder< index_type, 0, 0, 0 >,
            std::make_index_sequence< 3 >,
            Device > a, b;
   a.setSizes( size, size, size );
   b.setSizes( size, size, size );
   a.getStorageArray().setValue( -1 );
   b.getStorageArray().setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [&]() {
      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k ) mutable { a_view( i, j, k ) = b_view( i, j, k ); } );
      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k ) mutable { a_view( i, j, k ) = b_view( i, j, k ); } );
   };

   const double datasetSize = 2 * std::pow( size, 3 ) * sizeof(value_type) / oneGB;
   benchmark.setOperation( "3D", datasetSize );
   benchmark.time< Device >( reset, performer< Device >(), f );

   expect_eq( a.getStorageArray(), b.getStorageArray() );
}

// TODO: implement general ParallelBoundaryExecutor
//template< typename Device >
//void benchmark_4D( Benchmark& benchmark, index_type size = 150 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0 >,
//            std::make_index_sequence< 4 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size );
//   b.setSizes( size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 4 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "4D", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}
//
//template< typename Device >
//void benchmark_5D( Benchmark& benchmark, index_type size = 56 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0 >,
//            std::make_index_sequence< 5 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size );
//   b.setSizes( size, size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 5 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "5D", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}
//
//template< typename Device >
//void benchmark_6D( Benchmark& benchmark, index_type size = 28 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0, 0 >,
//            std::make_index_sequence< 6 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size, size );
//   b.setSizes( size, size, size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 6 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "6D", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}


template< typename Device >
void benchmark_2D_perm( Benchmark<>& benchmark, index_type size = 22333 )
{
   NDArray< value_type,
            SizesHolder< index_type, 0, 0 >,
            std::index_sequence< 1, 0 >,
            Device > a, b;
   a.setSizes( size, size );
   b.setSizes( size, size );
   a.getStorageArray().setValue( -1 );
   b.getStorageArray().setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [&]() {
      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j ) mutable { a_view( i, j ) = b_view( i, j ); } );
      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j ) mutable { a_view( i, j ) = b_view( i, j ); } );
   };

   const double datasetSize = 2 * std::pow( size, 2 ) * sizeof(value_type) / oneGB;
   benchmark.setOperation( "2D permuted", datasetSize );
   benchmark.time< Device >( reset, performer< Device >(), f );

   expect_eq( a.getStorageArray(), b.getStorageArray() );
}

template< typename Device >
void benchmark_3D_perm( Benchmark<>& benchmark, index_type size = 800 )
{
   NDArray< value_type,
            SizesHolder< index_type, 0, 0, 0 >,
            std::index_sequence< 2, 1, 0 >,
            Device > a, b;
   a.setSizes( size, size, size );
   b.setSizes( size, size, size );
   a.getStorageArray().setValue( -1 );
   b.getStorageArray().setValue( 1 );

   auto a_view = a.getView();
   auto b_view = b.getView();

   auto f = [&]() {
      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k ) mutable { a_view( i, j, k ) = b_view( i, j, k ); } );
      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k ) mutable { a_view( i, j, k ) = b_view( i, j, k ); } );
   };

   const double datasetSize = 2 * std::pow( size, 3 ) * sizeof(value_type) / oneGB;
   benchmark.setOperation( "3D permuted", datasetSize );
   benchmark.time< Device >( reset, performer< Device >(), f );

   expect_eq( a.getStorageArray(), b.getStorageArray() );
}

// TODO: implement general ParallelBoundaryExecutor
//template< typename Device >
//void benchmark_4D_perm( Benchmark& benchmark, index_type size = 150 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0 >,
//            std::index_sequence< 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size );
//   b.setSizes( size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l ) mutable { a_view( i, j, k, l ) = b_view( i, j, k, l ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 4 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "4D permuted", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}
//
//template< typename Device >
//void benchmark_5D_perm( Benchmark& benchmark, index_type size = 56 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0 >,
//            std::index_sequence< 4, 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size );
//   b.setSizes( size, size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m ) mutable { a_view( i, j, k, l, m ) = b_view( i, j, k, l, m ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 5 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "5D permuted", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}
//
//template< typename Device >
//void benchmark_6D_perm( Benchmark& benchmark, index_type size = 28 )
//{
//   NDArray< value_type,
//            SizesHolder< index_type, 0, 0, 0, 0, 0, 0 >,
//            std::index_sequence< 5, 4, 3, 2, 1, 0 >,
//            Device > a, b;
//   a.setSizes( size, size, size, size, size, size );
//   b.setSizes( size, size, size, size, size, size );
//   a.getStorageArray().setValue( -1 );
//   b.getStorageArray().setValue( 1 );
//
//   auto a_view = a.getView();
//   auto b_view = b.getView();
//
//   auto f = [&]() {
//      a.forBoundary( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } );
//      a.forInternal( [=] __cuda_callable__ ( index_type i, index_type j, index_type k, index_type l, index_type m, index_type n ) mutable { a_view( i, j, k, l, m, n ) = b_view( i, j, k, l, m, n ); } );
//   };
//
//   const double datasetSize = 2 * std::pow( size, 6 ) * sizeof(value_type) / oneGB;
//   benchmark.setOperation( "6D permuted", datasetSize );
//   benchmark.time< Device >( reset, performer< Device >(), f );
//
//   expect_eq( a.getStorageArray(), b.getStorageArray() );
//}

template< typename Device >
void run_benchmarks( Benchmark<>& benchmark )
{
   benchmark_1D< Device >( benchmark );
   benchmark_2D< Device >( benchmark );
   benchmark_3D< Device >( benchmark );
//   benchmark_4D< Device >( benchmark );
//   benchmark_5D< Device >( benchmark );
//   benchmark_6D< Device >( benchmark );
   benchmark_2D_perm< Device >( benchmark );
   benchmark_3D_perm< Device >( benchmark );
//   benchmark_4D_perm< Device >( benchmark );
//   benchmark_5D_perm< Device >( benchmark );
//   benchmark_6D_perm< Device >( benchmark );
}

void setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-ndarray-boundary.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
   #ifdef HAVE_CUDA
   config.addEntryEnum( "cuda" );
   #endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   const String devices = parameters.getParameter< String >( "devices" );
   if( devices == "all" || devices == "host" )
      run_benchmarks< Devices::Host >( benchmark );
#ifdef HAVE_CUDA
   if( devices == "all" || devices == "cuda" )
      run_benchmarks< Devices::Cuda >( benchmark );
#endif

   return EXIT_SUCCESS;
}
