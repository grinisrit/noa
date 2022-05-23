#include <sstream>
#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <sstream>

#include <TNL/zlib_compression.h>

#ifdef HAVE_GTEST
template< typename HeaderType >
void test_compress( std::string input, std::string expected_output )
{
   std::stringstream str;
   TNL::write_compressed_block< HeaderType >( input.c_str(), input.length(), str );
   const std::string output = str.str();
   EXPECT_EQ( output, expected_output );
}

template< typename HeaderType >
void test_decompress( std::string input, std::string expected_output )
{
   // decompress C string
   {
      const auto result = TNL::decompress_block< HeaderType, char >( input.c_str() );
      EXPECT_EQ( result.first, (HeaderType) expected_output.length() );
      const std::string output( result.second.get(), result.first );
      EXPECT_EQ( output, expected_output );
   }
   // decompress stream
   {
      std::stringstream str( input );
      const auto result = TNL::decompress_block< HeaderType, char >( str );
      EXPECT_EQ( result.first, (HeaderType) expected_output.length() );
      const std::string output( result.second.get(), result.first );
      EXPECT_EQ( output, expected_output );
   }
}

TEST( base64Test, compress_string )
{
   test_compress< std::int32_t > ( "hello, world", "AQAAAAwAAAAMAAAAFAAAAA==eJzLSM3JyddRKM8vykkBAB1UBIk=" );
   test_compress< std::uint32_t >( "hello, world", "AQAAAAwAAAAMAAAAFAAAAA==eJzLSM3JyddRKM8vykkBAB1UBIk=" );
   test_compress< std::int64_t > ( "hello, world", "AQAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAA=eJzLSM3JyddRKM8vykkBAB1UBIk=" );
   test_compress< std::uint64_t >( "hello, world", "AQAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAA=eJzLSM3JyddRKM8vykkBAB1UBIk=" );
}

TEST( base64Test, decode )
{
   test_decompress< std::int32_t  >( "AQAAAAwAAAAMAAAAFAAAAA==eJzLSM3JyddRKM8vykkBAB1UBIk=", "hello, world" );
   test_decompress< std::uint32_t >( "AQAAAAwAAAAMAAAAFAAAAA==eJzLSM3JyddRKM8vykkBAB1UBIk=", "hello, world" );
   test_decompress< std::int64_t  >( "AQAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAA=eJzLSM3JyddRKM8vykkBAB1UBIk=", "hello, world" );
   test_decompress< std::uint64_t >( "AQAAAAAAAAAMAAAAAAAAAAwAAAAAAAAAFAAAAAAAAAA=eJzLSM3JyddRKM8vykkBAB1UBIk=", "hello, world" );
}
#endif

#include "main.h"
