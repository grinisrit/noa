#include <iostream>
#include <TNL/String.h>

using namespace TNL;

int main()
{
    String names(  "       Josh Martin   John  Marley Charles   " );
    String names2( ".......Josh Martin...John..Marley.Charles..." );
    std::cout << "names strip is: " << names.strip() << std::endl;
    std::cout << "names2 strip is: " << names.strip( '.' ) << std::endl;
}
