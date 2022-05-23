#include <iostream>
#include <TNL/Logger.h>

using namespace TNL;
using namespace std;
       
int main()
{
    Logger logger(50,cout);
    
    logger.writeSystemInformation( false );

    logger.writeHeader("MyTitle");
    logger.writeSeparator();
    logger.writeSystemInformation( true );
    logger.writeSeparator();
}

