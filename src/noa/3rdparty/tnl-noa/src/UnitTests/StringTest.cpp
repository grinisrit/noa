// Implemented by Nina Dzugasova

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/String.h>
#include <TNL/File.h>

using namespace TNL;

static const char* TEST_FILE_NAME = "test_StringTest.tnl";

#ifdef HAVE_GTEST
TEST( StringTest, BasicConstructor )
{
   String str;
   EXPECT_STREQ( str.getString(), "" );
}

TEST( StringTest, CopyConstructor )
{
   String string( "string1" );
   String emptyString( "" );
   String string2( string );
   String emptyString2( emptyString );

   EXPECT_STREQ( string2.getString(), "string1" );
   EXPECT_STREQ( emptyString2.getString(), "" );
}

TEST( StringTest, convertToString )
{
   String string1 = convertToString( 10 );
   String string2 = convertToString( -5 );
   String string3 = convertToString( true );
   String string4 = convertToString( false );

   EXPECT_STREQ( string1.getString(), "10" );
   EXPECT_STREQ( string2.getString(), "-5" );
   EXPECT_STREQ( string3.getString(), "true" );
   EXPECT_STREQ( string4.getString(), "false" );
}

TEST( StringTest, GetSize )
{
    String str1( "string" );
    String str2( "12345" );
    String str3( "string3" );
    String str4( "String_4" );
    String str5( "Last String" );

    EXPECT_EQ( str1.getSize(), 6 );
    EXPECT_EQ( str2.getSize(), 5 );
    EXPECT_EQ( str3.getSize(), 7 );
    EXPECT_EQ( str4.getSize(), 8 );
    EXPECT_EQ( str5.getSize(), 11 );

    EXPECT_EQ( str1.getLength(), 6 );
    EXPECT_EQ( str2.getLength(), 5 );
    EXPECT_EQ( str3.getLength(), 7 );
    EXPECT_EQ( str4.getLength(), 8 );
    EXPECT_EQ( str5.getLength(), 11 );
}

TEST( StringTest, GetAllocatedSize )
{
    String str( "MeineKleine" );

    EXPECT_EQ( str.getLength(), 11 );
    EXPECT_GE( str.getAllocatedSize(), str.getLength() );
}

TEST( StringTest, SetSize )
{
   String str;
   str.setSize( 42 );
   EXPECT_EQ( str.getSize(), 42 );
   EXPECT_GE( str.getAllocatedSize(), 42 );
}

TEST( StringTest, GetString )
{
    String str( "MyString" );
    EXPECT_EQ( strcmp( str.getString(), "MyString" ), 0 );
}

TEST( StringTest, IndexingOperator )
{
   String str( "1234567890" );
   EXPECT_EQ( str[ 0 ], '1' );
   EXPECT_EQ( str[ 1 ], '2' );
   EXPECT_EQ( str[ 2 ], '3' );
   EXPECT_EQ( str[ 3 ], '4' );
   EXPECT_EQ( str[ 4 ], '5' );
   EXPECT_EQ( str[ 5 ], '6' );
   EXPECT_EQ( str[ 6 ], '7' );
   EXPECT_EQ( str[ 7 ], '8' );
   EXPECT_EQ( str[ 8 ], '9' );
   EXPECT_EQ( str[ 9 ], '0' );
}

TEST( StringTest, CStringOperators )
{
   // assignment operator
   String string1;
   string1 = "string";
   EXPECT_STREQ( string1.getString(), "string" );

   // addition
   string1 += "string2";
   EXPECT_STREQ( string1.getString(), "stringstring2" );

   // addition that forces a new page allocation
   string1 += " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long";
   EXPECT_STREQ( string1.getString(),
              "stringstring2"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
              " long long long long long long long long long long long long long long"
            );

   // addition
   EXPECT_STREQ( (String( "foo " ) + "bar").getString(), "foo bar" );
   EXPECT_STREQ( ("foo" + String( " bar" )).getString(), "foo bar" );

   // comparison
   EXPECT_EQ( String( "foo" ), "foo" );
   EXPECT_NE( String( "bar" ), "foo" );
   EXPECT_NE( String( "fooo" ), "foo" );
}

TEST( StringTest, StringOperators )
{
   // assignment
   String string1( "string" );
   String string2;
   string2 = string1;
   EXPECT_STREQ( string2.getString(), "string" );

   // addition
   string1 = "foo ";
   string1 += String( "bar" );
   EXPECT_STREQ( string1.getString(), "foo bar" );

   // comparison
   EXPECT_EQ( String( "foo bar" ), string1 );
   EXPECT_NE( String( "bar" ), string1 );
   EXPECT_NE( String( "bar" ), String( "baz" ) );
   EXPECT_NE( String( "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long" ),
              String( "short" ) );
   String string3( "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long "
                   "long long long long long long long long long long long" );
   string3[ 255 ] = 0;
   // std::string knows the original length
   EXPECT_NE( string3,
              String( "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long " ) );
   // C string can be terminated in the middle
   EXPECT_STREQ( string3.getString(),
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long long long long long "
                      "long long long long long long long " );

   // addition
   EXPECT_EQ( String( "foo " ) + String( "bar" ), "foo bar" );
}

TEST( StringTest, SingleCharacterOperators )
{
   // assignment
   String string1;
   string1 = 'A';
   EXPECT_STREQ( string1.getString(), "A" );

   // addition of a single character
   String string2( "string " );
   string2 += 'A';
   EXPECT_STREQ( string2.getString(), "string A" );

   // addition of a single character that causes new page allocation
   string2 = "long long long long long long long long long long long long long "
             "long long long long long long long long long long long long long "
             "long long long long long long long long long long long long long "
             "long long long long long long long long long long long long ";
   ASSERT_EQ( string2.getLength(), 255 );
   string2 += 'B';
   EXPECT_STREQ( string2.getString(),
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long long "
                  "long long long long long long long long long long long long B"
               );

   // addition
   EXPECT_STREQ( (String( "A " ) + 'B').getString(), "A B" );
   EXPECT_STREQ( ('A' + String( " B" )).getString(), "A B" );

   // comparison
   EXPECT_EQ( String( "A" ), 'A' );
   EXPECT_NE( String( "B" ), 'A' );
   EXPECT_NE( String( "AB" ), 'A' );
}

TEST( StringTest, CastToBoolOperator )
{
   String string;
   EXPECT_TRUE( ! string );
   EXPECT_FALSE( string );
   string = "foo";
   EXPECT_TRUE( string );
   EXPECT_FALSE( ! string );
}

TEST( StringTest, replace )
{
   EXPECT_EQ( String( "string" ).replace( "ing", "bc" ), "strbc" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT" ), "CATracadCATra" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT", 1 ), "CATracadabra" );
   EXPECT_NE( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "abracadCATra" );
   EXPECT_NE( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "abracadabra" );
   EXPECT_EQ( String( "abracadabra" ).replace( "ab", "CAT", 2 ), "CATracadCATra" );
}

TEST( StringTest, strip )
{
   EXPECT_EQ( String( "string" ).strip(), "string" );
   EXPECT_EQ( String( "  string" ).strip(), "string" );
   EXPECT_EQ( String( "string  " ).strip(), "string" );
   EXPECT_EQ( String( "  string  " ).strip(), "string" );
   EXPECT_EQ( String( " string1  string2  " ).strip(), "string1  string2" );
   EXPECT_EQ( String( "" ).strip(), "" );
   EXPECT_EQ( String( "  " ).strip(), "" );
}

TEST( StringTest, split )
{
   std::vector< String > parts;

   parts = String( "A B C" ).split( ' ' );
   ASSERT_EQ( (int) parts.size(), 3 );
   EXPECT_EQ( parts[ 0 ], "A" );
   EXPECT_EQ( parts[ 1 ], "B" );
   EXPECT_EQ( parts[ 2 ], "C" );

   parts = String( "abracadabra" ).split( 'a' );
   ASSERT_EQ( (int) parts.size(), 6 );
   EXPECT_EQ( parts[ 0 ], "" );
   EXPECT_EQ( parts[ 1 ], "br" );
   EXPECT_EQ( parts[ 2 ], "c" );
   EXPECT_EQ( parts[ 3 ], "d" );
   EXPECT_EQ( parts[ 4 ], "br" );
   EXPECT_EQ( parts[ 5 ], "" );

   parts = String( "abracadabra" ).split( 'a', String::SplitSkip::SkipEmpty );
   ASSERT_EQ( (int) parts.size(), 4 );
   EXPECT_EQ( parts[ 0 ], "br" );
   EXPECT_EQ( parts[ 1 ], "c" );
   EXPECT_EQ( parts[ 2 ], "d" );
   EXPECT_EQ( parts[ 3 ], "br" );

   parts = String( "abracadabra" ).split( 'b' );
   ASSERT_EQ( (int) parts.size(), 3 );
   EXPECT_EQ( parts[ 0 ], "a" );
   EXPECT_EQ( parts[ 1 ], "racada" );
   EXPECT_EQ( parts[ 2 ], "ra" );

   parts = String( "abracadabra" ).split( 'A' );
   ASSERT_EQ( (int) parts.size(), 1 );
   EXPECT_EQ( parts[ 0 ], "abracadabra" );

   parts = String( "a,,b,c" ).split( ',' );
   ASSERT_EQ( (int) parts.size(), 4 );
   EXPECT_EQ( parts[ 0 ], "a" );
   EXPECT_EQ( parts[ 1 ], "" );
   EXPECT_EQ( parts[ 2 ], "b" );
   EXPECT_EQ( parts[ 3 ], "c" );
}

TEST( StringTest, SaveLoad )
{
   String str1( "testing-string" );
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );
   ASSERT_NO_THROW( file << str1 );
   ASSERT_NO_THROW( file.close() );
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   String str2;
   ASSERT_NO_THROW( file >> str2 );
   EXPECT_EQ( str1, str2 );
   ASSERT_NO_THROW( file.close() );
   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TEST( StringTest, startsWith )
{
   String str( "abracadabra" );
   EXPECT_TRUE( str.startsWith( "a" ) );
   EXPECT_TRUE( str.startsWith( "ab" ) );
   EXPECT_TRUE( str.startsWith( "abr" ) );
   EXPECT_TRUE( str.startsWith( "abra" ) );
   EXPECT_TRUE( str.startsWith( "abrac" ) );
   EXPECT_TRUE( str.startsWith( "abraca" ) );
   EXPECT_TRUE( str.startsWith( "abracad" ) );
   EXPECT_TRUE( str.startsWith( "abracada" ) );
   EXPECT_TRUE( str.startsWith( "abracadab" ) );
   EXPECT_TRUE( str.startsWith( "abracadabr" ) );
   EXPECT_TRUE( str.startsWith( "abracadabra" ) );
   EXPECT_FALSE( str.startsWith( "b" ) );
   EXPECT_FALSE( str.startsWith( "aa" ) );
   EXPECT_FALSE( str.startsWith( "aba" ) );
   EXPECT_FALSE( str.startsWith( "abrb" ) );
   EXPECT_FALSE( str.startsWith( "abrad" ) );
   EXPECT_FALSE( str.startsWith( "abracb" ) );
   EXPECT_FALSE( str.startsWith( "abracaa" ) );
   EXPECT_FALSE( str.startsWith( "abracadb" ) );
   EXPECT_FALSE( str.startsWith( "abracadaa" ) );
   EXPECT_FALSE( str.startsWith( "abracadaba" ) );
   EXPECT_FALSE( str.startsWith( "abracadabrb" ) );
   EXPECT_FALSE( str.startsWith( "abracadabrab" ) );
}

TEST( StringTest, endsWith )
{
   String str( "abracadabra" );
   EXPECT_TRUE( str.endsWith( "a" ) );
   EXPECT_TRUE( str.endsWith( "ra" ) );
   EXPECT_TRUE( str.endsWith( "bra" ) );
   EXPECT_TRUE( str.endsWith( "abra" ) );
   EXPECT_TRUE( str.endsWith( "dabra" ) );
   EXPECT_TRUE( str.endsWith( "adabra" ) );
   EXPECT_TRUE( str.endsWith( "cadabra" ) );
   EXPECT_TRUE( str.endsWith( "acadabra" ) );
   EXPECT_TRUE( str.endsWith( "racadabra" ) );
   EXPECT_TRUE( str.endsWith( "bracadabra" ) );
   EXPECT_TRUE( str.endsWith( "abracadabra" ) );
   EXPECT_FALSE( str.endsWith( "b" ) );
   EXPECT_FALSE( str.endsWith( "ba" ) );
   EXPECT_FALSE( str.endsWith( "ara" ) );
   EXPECT_FALSE( str.endsWith( "bbra" ) );
   EXPECT_FALSE( str.endsWith( "babra" ) );
   EXPECT_FALSE( str.endsWith( "bdabra" ) );
   EXPECT_FALSE( str.endsWith( "badabra" ) );
   EXPECT_FALSE( str.endsWith( "bcadabra" ) );
   EXPECT_FALSE( str.endsWith( "aacadabra" ) );
   EXPECT_FALSE( str.endsWith( "aracadabra" ) );
   EXPECT_FALSE( str.endsWith( "bbracadabra" ) );
   EXPECT_FALSE( str.endsWith( "babracadabra" ) );
}
#endif

#include "main.h"
