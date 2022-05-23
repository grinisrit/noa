// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Grid.h"

void export_Grid1D( py::module & m )
{
    export_Grid< Grid1D >( m, "Grid1D" );
}
