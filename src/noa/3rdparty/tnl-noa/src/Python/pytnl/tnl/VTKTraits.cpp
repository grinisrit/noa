#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/Meshes/VTKTraits.h>

void export_VTKTraits( py::module & m )
{
    py::enum_< TNL::Meshes::VTK::FileFormat >( m, "VTKFileFormat")
       .value("ascii", TNL::Meshes::VTK::FileFormat::ascii)
       .value("binary", TNL::Meshes::VTK::FileFormat::binary)
       .value("zlib_compressed", TNL::Meshes::VTK::FileFormat::zlib_compressed)
    ;
    py::enum_< TNL::Meshes::VTK::DataType >( m, "VTKDataType")
       .value("CellData", TNL::Meshes::VTK::DataType::CellData)
       .value("PointData", TNL::Meshes::VTK::DataType::PointData)
    ;
    py::enum_< TNL::Meshes::VTK::EntityShape >( m, "VTKEntityShape")
       .value("Vertex", TNL::Meshes::VTK::EntityShape::Vertex)
       .value("PolyVertex", TNL::Meshes::VTK::EntityShape::PolyVertex)
       .value("Line", TNL::Meshes::VTK::EntityShape::Line)
       .value("PolyLine", TNL::Meshes::VTK::EntityShape::PolyLine)
       .value("Triangle", TNL::Meshes::VTK::EntityShape::Triangle)
       .value("TriangleStrip", TNL::Meshes::VTK::EntityShape::TriangleStrip)
       .value("Polygon", TNL::Meshes::VTK::EntityShape::Polygon)
       .value("Pixel", TNL::Meshes::VTK::EntityShape::Pixel)
       .value("Quad", TNL::Meshes::VTK::EntityShape::Quad)
       .value("Tetra", TNL::Meshes::VTK::EntityShape::Tetra)
       .value("Voxel", TNL::Meshes::VTK::EntityShape::Voxel)
       .value("Hexahedron", TNL::Meshes::VTK::EntityShape::Hexahedron)
       .value("Wedge", TNL::Meshes::VTK::EntityShape::Wedge)
       .value("Pyramid", TNL::Meshes::VTK::EntityShape::Pyramid)
    ;
    py::enum_< TNL::Meshes::VTK::CellGhostTypes >( m, "VTKCellGhostTypes")
       .value("DUPLICATECELL", TNL::Meshes::VTK::CellGhostTypes::DUPLICATECELL,               "the cell is present on multiple processors")
       .value("HIGHCONNECTIVITYCELL", TNL::Meshes::VTK::CellGhostTypes::HIGHCONNECTIVITYCELL, "the cell has more neighbors than in a regular mesh")
       .value("LOWCONNECTIVITYCELL", TNL::Meshes::VTK::CellGhostTypes::LOWCONNECTIVITYCELL,   "the cell has less neighbors than in a regular mesh")
       .value("REFINEDCELL", TNL::Meshes::VTK::CellGhostTypes::REFINEDCELL,                   "other cells are present that refines it")
       .value("EXTERIORCELL", TNL::Meshes::VTK::CellGhostTypes::EXTERIORCELL,                 "the cell is on the exterior of the data set")
       .value("HIDDENCELL", TNL::Meshes::VTK::CellGhostTypes::HIDDENCELL,                     "the cell is needed to maintain connectivity, but the data values should be ignored")
    ;
    py::enum_< TNL::Meshes::VTK::PointGhostTypes >( m, "VTKPointGhostTypes")
       .value("DUPLICATEPOINT", TNL::Meshes::VTK::PointGhostTypes::DUPLICATEPOINT, "the cell is present on multiple processors")
       .value("HIDDENPOINT", TNL::Meshes::VTK::PointGhostTypes::HIDDENPOINT,       "the point is needed to maintain connectivity, but the data values should be ignored")
    ;
}
