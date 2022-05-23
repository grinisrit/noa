#include <iostream>
#include <TNL/Object.h>

using namespace TNL;

int main()
{
   auto parsedObjectType = parseObjectType( String( "MyObject< Value, Device, Index >" ) );
   for( auto &token : parsedObjectType )
      std::cout << token << std::endl;
}
