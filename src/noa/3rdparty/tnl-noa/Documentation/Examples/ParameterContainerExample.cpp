#include <iostream>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/String.h>

using namespace TNL;

int main()
{
    Config::ParameterContainer parameters;
    String param = parameters.getParameter< String >( "distributed-grid-io-type" );
//    parameters.checkParameter< String >( "distributed-grid-io-type" );
}
