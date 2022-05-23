#include <iostream>
#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <unistd.h>

using namespace TNL;

int main()
{
    unsigned int microseconds = 0.5e6;
    Timer time;
    time.start();
    usleep(microseconds);
    time.stop();

    Logger logger( 50, std::cout );
    time.writeLog( logger, 0 );
}
