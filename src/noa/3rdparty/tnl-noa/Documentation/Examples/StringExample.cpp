#include <iostream>
#include <TNL/String.h>
#include <TNL/File.h>

using namespace TNL;

int main( int argc, char* argv[] )
{
   String emptyString;
   String string1( "string 1" );
   String string2( "string 2" );
   String string3( string2 );
   String string4 = convertToString( 28.4 );

   std::cout << "empytString = " << emptyString << std::endl;
   std::cout << "string1 = " << string1 << std::endl;
   std::cout << "string2 = " << string2 << std::endl;
   std::cout << "string3 = " << string3 << std::endl;
   std::cout << "string4 = " << string4 << std::endl;

   std::cout << "emptyString size = " << emptyString.getSize() << std::endl;
   std::cout << "string1 size = " << string1.getSize() << std::endl;
   std::cout << "string1 length = " << string1.getLength() << std::endl;

   const char* c_string = string1.getString();
   std::cout << "c_string = " << c_string << std::endl;

   std::cout << " 3rd letter of string1 =" << string1[ 2 ] << std::endl;

   std::cout << " string1 + string2 = " << string1 + string2 << std::endl;
   std::cout << " string1 + \" another string\" = " << string1 + " another string" << std::endl;

   string2 += "another string";
   std::cout << " string2 = " << string2;
   string2 = "string 2";

   if( string3 == string2 )
      std::cout << "string3 == string2" << std::endl;
   if( string1 != string2 )
      std::cout << "string1 != string2" << std::endl;

   if( ! emptyString )
      std::cout << "emptyString is empty" << std::endl;
   if( string1 )
      std::cout << "string1 is not empty" << std::endl;

   File myFile;
   myFile.open( "string_save.out", std::ios_base::out );
   myFile << string1;
   myFile.close();

   myFile.open( "string_save.out", std::ios_base::in );
   myFile >> string3;
   std::cout << "string 3 after loading = " << string3 << std::endl;
}
