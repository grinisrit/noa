// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Grid.h"

void export_Grid3D( py::module & m )
{
    export_Grid< Grid3D >( m, "Grid3D" );
}
