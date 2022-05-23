// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Grid.h"

void export_Grid2D( py::module & m )
{
    export_Grid< Grid2D >( m, "Grid2D" );
}
