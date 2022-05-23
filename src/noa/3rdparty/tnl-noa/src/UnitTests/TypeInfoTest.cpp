#include <TNL/TypeInfo.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST

enum MyEnumType { foo, bar };
enum class MyEnumClass { foo, bar };

class MyClass {};
class MyClassWithGetSerializationType
{
public:
   static std::string getSerializationType() { return "SomethingElse"; }
};

template< typename... >
class MyClassTemplate {};

class MyPolymorphicBase
{
public:
   virtual ~MyPolymorphicBase() {}
};
class MyPolymorphicDerived : public MyPolymorphicBase
{
public:
   virtual ~MyPolymorphicDerived() {}
};


TEST( TypeInfoTest, getType )
{
   // non-const variants
   EXPECT_EQ( getType< void >(), std::string( "void" ) );
   EXPECT_EQ( getType< bool >(), std::string( "bool" ) );

   EXPECT_EQ( getType< char >(), std::string( "char" ) );
   EXPECT_EQ( getType< short >(), std::string( "short" ) );
   EXPECT_EQ( getType< int >(), std::string( "int" ) );
   EXPECT_EQ( getType< long >(), std::string( "long" ) );

   EXPECT_EQ( getType< unsigned char >(), std::string( "unsigned char" ) );
   EXPECT_EQ( getType< unsigned short >(), std::string( "unsigned short" ) );
   EXPECT_EQ( getType< unsigned int >(), std::string( "unsigned int" ) );
   EXPECT_EQ( getType< unsigned long >(), std::string( "unsigned long" ) );

   EXPECT_EQ( getType< signed char >(), std::string( "signed char" ) );

   EXPECT_EQ( getType< float >(), std::string( "float" ) );
   EXPECT_EQ( getType< double >(), std::string( "double" ) );
   EXPECT_EQ( getType< long double >(), std::string( "long double" ) );

   // const variants - top-level cv-qualifiers are ignored
   EXPECT_EQ( getType< const void >(), std::string( "void" ) );
   EXPECT_EQ( getType< const bool >(), std::string( "bool" ) );

   EXPECT_EQ( getType< const char >(), std::string( "char" ) );
   EXPECT_EQ( getType< const short >(), std::string( "short" ) );
   EXPECT_EQ( getType< const int >(), std::string( "int" ) );
   EXPECT_EQ( getType< const long >(), std::string( "long" ) );

   EXPECT_EQ( getType< const unsigned char >(), std::string( "unsigned char" ) );
   EXPECT_EQ( getType< const unsigned short >(), std::string( "unsigned short" ) );
   EXPECT_EQ( getType< const unsigned int >(), std::string( "unsigned int" ) );
   EXPECT_EQ( getType< const unsigned long >(), std::string( "unsigned long" ) );

   EXPECT_EQ( getType< const signed char >(), std::string( "signed char" ) );

   EXPECT_EQ( getType< const float >(), std::string( "float" ) );
   EXPECT_EQ( getType< const double >(), std::string( "double" ) );
   EXPECT_EQ( getType< const long double >(), std::string( "long double" ) );

   // enum types
   EXPECT_EQ( getType< MyEnumType >(), std::string( "MyEnumType" ) );
   EXPECT_EQ( getType< MyEnumClass >(), std::string( "MyEnumClass" ) );

   // classes
   EXPECT_EQ( getType< MyClass >(), std::string( "MyClass" ) );
   EXPECT_EQ( getType< MyClassWithGetSerializationType >(), std::string( "MyClassWithGetSerializationType" ) );

   // class templates
   using T1 = MyClassTemplate< int, MyClassTemplate< int, int >, MyClass >;
   EXPECT_EQ( getType< T1 >(), std::string( "MyClassTemplate<int, MyClassTemplate<int, int>, MyClass>" ) );

   // polymorphic base
   MyPolymorphicDerived obj;
   MyPolymorphicBase* ptr = &obj;
   // no dynamic cast for pointer types
   EXPECT_EQ( getType( ptr ), std::string( "MyPolymorphicBase*" ) );
   // reference to a polymorphic object gets dynamic cast
   EXPECT_EQ( getType( *ptr ), std::string( "MyPolymorphicDerived" ) );
}

TEST( TypeInfoTest, getSerializationType )
{
   // non-const variants
   EXPECT_EQ( getSerializationType< void >(), std::string( "void" ) );
   EXPECT_EQ( getSerializationType< bool >(), std::string( "bool" ) );

   EXPECT_EQ( getSerializationType< char >(), std::string( "char" ) );
   EXPECT_EQ( getSerializationType< short >(), std::string( "short" ) );
   EXPECT_EQ( getSerializationType< int >(), std::string( "int" ) );
   EXPECT_EQ( getSerializationType< long >(), std::string( "long" ) );

   EXPECT_EQ( getSerializationType< unsigned char >(), std::string( "unsigned char" ) );
   EXPECT_EQ( getSerializationType< unsigned short >(), std::string( "unsigned short" ) );
   EXPECT_EQ( getSerializationType< unsigned int >(), std::string( "unsigned int" ) );
   EXPECT_EQ( getSerializationType< unsigned long >(), std::string( "unsigned long" ) );

   EXPECT_EQ( getSerializationType< signed char >(), std::string( "signed char" ) );

   EXPECT_EQ( getSerializationType< float >(), std::string( "float" ) );
   EXPECT_EQ( getSerializationType< double >(), std::string( "double" ) );
   EXPECT_EQ( getSerializationType< long double >(), std::string( "long double" ) );

   // const variants - top-level cv-qualifiers are ignored
   EXPECT_EQ( getSerializationType< const void >(), std::string( "void" ) );
   EXPECT_EQ( getSerializationType< const bool >(), std::string( "bool" ) );

   EXPECT_EQ( getSerializationType< const char >(), std::string( "char" ) );
   EXPECT_EQ( getSerializationType< const short >(), std::string( "short" ) );
   EXPECT_EQ( getSerializationType< const int >(), std::string( "int" ) );
   EXPECT_EQ( getSerializationType< const long >(), std::string( "long" ) );

   EXPECT_EQ( getSerializationType< const unsigned char >(), std::string( "unsigned char" ) );
   EXPECT_EQ( getSerializationType< const unsigned short >(), std::string( "unsigned short" ) );
   EXPECT_EQ( getSerializationType< const unsigned int >(), std::string( "unsigned int" ) );
   EXPECT_EQ( getSerializationType< const unsigned long >(), std::string( "unsigned long" ) );

   EXPECT_EQ( getSerializationType< const signed char >(), std::string( "signed char" ) );

   EXPECT_EQ( getSerializationType< const float >(), std::string( "float" ) );
   EXPECT_EQ( getSerializationType< const double >(), std::string( "double" ) );
   EXPECT_EQ( getSerializationType< const long double >(), std::string( "long double" ) );

   // enum types
   EXPECT_EQ( getSerializationType< MyEnumType >(), std::string( "MyEnumType" ) );
   EXPECT_EQ( getSerializationType< MyEnumClass >(), std::string( "MyEnumClass" ) );

   // classes
   EXPECT_EQ( getSerializationType< MyClass >(), std::string( "MyClass" ) );
   EXPECT_EQ( getSerializationType< MyClassWithGetSerializationType >(), std::string( "SomethingElse" ) );

   // class templates
   using T1 = MyClassTemplate< int, MyClassTemplate< int, int >, MyClass >;
   EXPECT_EQ( getSerializationType< T1 >(), "MyClassTemplate<int, MyClassTemplate<int, int>, MyClass>" );
}
#endif

#include "main.h"
