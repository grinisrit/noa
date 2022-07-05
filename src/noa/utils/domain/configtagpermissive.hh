#pragma once

template <typename CellTopology> struct ConfigTagPermissive {};

namespace noa::TNL::Meshes::BuildConfigTags {
        template <typename CellTopology>
        struct MeshCellTopologyTag<ConfigTagPermissive<CellTopology>, CellTopology> { enum { enabled = true }; };
}
