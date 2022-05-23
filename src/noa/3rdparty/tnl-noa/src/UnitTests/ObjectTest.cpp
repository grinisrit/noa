#include <TNL/Devices/Host.h>
#include <TNL/Object.h>
#include <TNL/File.h>
#include <TNL/Containers/Array.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

static const char* TEST_FILE_NAME = "test_ObjectTest.tnl";

#ifdef HAVE_GTEST
TEST( ObjectTest, SaveAndLoadTest )
{
   Object testObject;
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );
   ASSERT_NO_THROW( testObject.save( file ) );
   ASSERT_NO_THROW( file.close() );
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   ASSERT_NO_THROW( testObject.load( file ) );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TEST( ObjectTest, parseObjectTypeTest )
{
   std::vector< String > parsed;
   std::vector< String > expected;

   // plain type
   parsed = parseObjectType( "int" );
   expected = {"int"};
   EXPECT_EQ( parsed, expected );

   // type with space
   parsed = parseObjectType( "short int" );
   expected = {"short int"};
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "unsigned short int" );
   expected = {"unsigned short int"};
   EXPECT_EQ( parsed, expected );

   // composed type
   parsed = parseObjectType( "Containers::Vector< double, Devices::Host, int >" );
   expected = { "Containers::Vector", "double", "Devices::Host", "int" };
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "Containers::Vector< Containers::List< String >, Devices::Host, int >" );
   expected = { "Containers::Vector", "Containers::List< String >", "Devices::Host", "int" };
   EXPECT_EQ( parsed, expected );

   // spaces in the template parameter
   parsed = parseObjectType( "A< short int >" );
   expected = { "A", "short int" };
   EXPECT_EQ( parsed, expected );

   parsed = parseObjectType( "A< B< short int >, C >" );
   expected = { "A", "B< short int >", "C" };
   EXPECT_EQ( parsed, expected );

   // spaces at different places in the template parameter
   parsed = parseObjectType( "A< b , c <E>  ,d>" );
   expected = { "A", "b", "c <E>", "d" };
   EXPECT_EQ( parsed, expected );
}

TEST( HeaderTest, SaveAndLoadTest )
{
   Object testObject;
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );
   ASSERT_NO_THROW( saveObjectType( file, "TYPE" ) );
   ASSERT_NO_THROW( file.close() );
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   String type;
   ASSERT_NO_THROW( type = getObjectType( file ) );
   EXPECT_EQ( type, "TYPE" );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}
#endif

#include "main.h"
