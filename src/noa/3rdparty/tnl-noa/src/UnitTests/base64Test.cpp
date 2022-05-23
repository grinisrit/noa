#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/base64.h>

using namespace TNL::base64;

#ifdef HAVE_GTEST
TEST( base64Test, get_encoded_length )
{
   EXPECT_EQ( get_encoded_length( 0 ), 0UL );
   EXPECT_EQ( get_encoded_length( 1 ), 4UL );
   EXPECT_EQ( get_encoded_length( 2 ), 4UL );
   EXPECT_EQ( get_encoded_length( 3 ), 4UL );
   EXPECT_EQ( get_encoded_length( 4 ), 8UL );
   EXPECT_EQ( get_encoded_length( 5 ), 8UL );
   EXPECT_EQ( get_encoded_length( 6 ), 8UL );
   EXPECT_EQ( get_encoded_length( 7 ), 12UL );
   EXPECT_EQ( get_encoded_length( 8 ), 12UL );
   EXPECT_EQ( get_encoded_length( 9 ), 12UL );
}

void test_encode_block( std::string input, std::string expected_output )
{
   const auto result = encode( (const std::uint8_t*) input.c_str(), input.length() );
   const std::string encoded( result.get() );
   EXPECT_EQ( encoded.length(), get_encoded_length( input.length() ) );
   EXPECT_EQ( encoded, expected_output );
}

void test_decode_block( std::string input, std::string expected_output )
{
   const auto result = decode( input.c_str(), input.length() );
   EXPECT_EQ( result.first, expected_output.length() );
   const std::string decoded( reinterpret_cast<const char*>(result.second.get()), result.first );
   EXPECT_EQ( decoded, expected_output );
}

TEST( base64Test, encode )
{
   test_encode_block( "hello, world", "aGVsbG8sIHdvcmxk" );
   test_encode_block( "hello, worl",  "aGVsbG8sIHdvcmw=" );
   test_encode_block( "hello, wor",   "aGVsbG8sIHdvcg==" );
   test_encode_block( "hello, wo",    "aGVsbG8sIHdv"     );
   test_encode_block( "hello, w",     "aGVsbG8sIHc="     );
   test_encode_block( "hello, ",      "aGVsbG8sIA=="     );
   test_encode_block( "hello,",       "aGVsbG8s"         );
   test_encode_block( "hello",        "aGVsbG8="         );
   test_encode_block( "hell",         "aGVsbA=="         );
   test_encode_block( "hel",          "aGVs"             );
   test_encode_block( "he",           "aGU="             );
   test_encode_block( "h",            "aA=="             );
   test_encode_block( "",             ""                 );
}

TEST( base64Test, decode )
{
   test_decode_block( "aGVsbG8sIHdvcmxk", "hello, world" );
   test_decode_block( "aGVsbG8sIHdvcmw=", "hello, worl"  );
   test_decode_block( "aGVsbG8sIHdvcg==", "hello, wor"   );
   test_decode_block( "aGVsbG8sIHdv",     "hello, wo"    );
   test_decode_block( "aGVsbG8sIHc=",     "hello, w"     );
   test_decode_block( "aGVsbG8sIA==",     "hello, "      );
   test_decode_block( "aGVsbG8s",         "hello,"       );
   test_decode_block( "aGVsbG8=",         "hello"        );
   test_decode_block( "aGVsbA==",         "hell"         );
   test_decode_block( "aGVs",             "hel"          );
   test_decode_block( "aGU=",             "he"           );
   test_decode_block( "aA==",             "h"            );
   test_decode_block( "",                 ""             );
}

TEST( base64Test, decode_invalid_chars )
{
   test_decode_block( "_", "" );
   test_decode_block( "a_A==", "h" );
   test_decode_block( "aA_==", "h" );
   test_decode_block( "aA=_=", "h" );
   test_decode_block( "aA==_", "h" );
}

TEST( base64Test, decode_invalid_padding )
{
   EXPECT_THROW( test_decode_block( "aaa", "" ),  std::invalid_argument );
   EXPECT_THROW( test_decode_block( "aa", "" ),   std::invalid_argument );
   EXPECT_THROW( test_decode_block( "a", "" ),    std::invalid_argument );
   EXPECT_THROW( test_decode_block( "aa=", "" ),  std::invalid_argument );
   EXPECT_THROW( test_decode_block( "a===", "" ), std::invalid_argument );
   EXPECT_THROW( test_decode_block( "a==", "" ),  std::invalid_argument );
   EXPECT_THROW( test_decode_block( "a=", "" ),   std::invalid_argument );
   EXPECT_THROW( test_decode_block( "=", "" ),    std::invalid_argument );
   EXPECT_THROW( test_decode_block( "==", "" ),   std::invalid_argument );
   EXPECT_THROW( test_decode_block( "===", "" ),  std::invalid_argument );
   EXPECT_THROW( test_decode_block( "====", "" ), std::invalid_argument );
}

TEST( base64Test, decode_ignore_after_padding )
{
   test_decode_block( "aGVsbG8=x", "hello" );
   test_decode_block( "aGVsbG8=xx", "hello" );
   test_decode_block( "aGVsbG8=xxx", "hello" );
   test_decode_block( "aGVsbG8=xxxx", "hello" );
   test_decode_block( "aGVsbG8=xxxxx", "hello" );
   test_decode_block( "aGVsbG8=xxxxxx", "hello" );
   test_decode_block( "aGVsbG8=xxxxxxx", "hello" );
   test_decode_block( "aGVsbG8=xxxxxxxx", "hello" );
}
#endif

#include "main.h"
