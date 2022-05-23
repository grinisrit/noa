#include <iostream>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/String.h>

using namespace TNL;
using namespace std;

int main()
{
    Config::ConfigDescription confd;
    confd.template addEntry< String >("--new-entry","Specific description.");
    confd.template addEntryEnum< String >("option1");
    confd.template addEntryEnum< String >("option2");
    confd.addDelimiter("-----------------------------");
}
