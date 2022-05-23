#include <iostream>
#include <TNL/Timer.h>

using namespace TNL;

int main()
{
    unsigned int microseconds = 0.5e6;
    Timer time;
    time.start();
    usleep(microseconds);
    time.stop();
    std::cout << "Elapsed real time: " << time.getRealTime() << std::endl;
    std::cout << "Elapsed CPU time: " << time.getCPUTime() << std::endl;
    std::cout << "Elapsed CPU cycles: " << time.getCPUCycles() << std::endl;
    time.reset();
    std::cout << "Real time after reset:" << time.getRealTime() << std::endl;
    std::cout << "CPU time after reset: " << time.getCPUTime() << std::endl;
    std::cout << "CPU cycles after reset: " << time.getCPUCycles() << std::endl;
}
