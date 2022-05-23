#include <iostream>
#include <TNL/String.h>

using namespace TNL;

int main()
{
    String str("my world");
    std::cout << "Allocated_size = " << str.getAllocatedSize() << std::endl;
}
