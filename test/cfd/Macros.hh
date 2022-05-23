#pragma once

// These will be used a lot in the files and take a lot of space
#define DOMAIN_TARGS	typename CellTopology, typename Device, typename Real, typename GlobalIndex, typename LocalIndex
#define DOMAIN_TYPE	Storage::Domain<CellTopology, Device, Real, GlobalIndex, LocalIndex>
