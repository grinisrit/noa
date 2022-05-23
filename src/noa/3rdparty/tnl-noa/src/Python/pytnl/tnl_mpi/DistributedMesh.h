#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

template< typename Mesh >
void export_DistributedMesh( py::module & m, const char* name )
{
    auto mesh = py::class_< Mesh >( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Mesh::getMeshDimension)
//        .def("setmunicationGroup", &Mesh::setCommunicationGroup)
//        .def("getmunicationGroup", &Mesh::getCommunicationGroup)
        .def("getLocalMesh", py::overload_cast<>(&Mesh::getLocalMesh), py::return_value_policy::reference_internal)
        .def("setGhostLevels", &Mesh::setGhostLevels)
        .def("getGhostLevels", &Mesh::getGhostLevels)
        .def("getGlobalPointIndices", []( const Mesh& mesh ) -> typename Mesh::GlobalIndexArray const& {
                return mesh.template getGlobalIndices< 0 >();
            },
            py::return_value_policy::reference_internal)
        .def("getGlobalCellIndices", []( const Mesh& mesh ) -> typename Mesh::GlobalIndexArray const& {
                return mesh.template getGlobalIndices< Mesh::getMeshDimension() >();
            },
            py::return_value_policy::reference_internal)
        .def("vtkPointGhostTypes", []( const Mesh& mesh ) -> typename Mesh::VTKTypesArrayType const& {
                return mesh.vtkPointGhostTypes();
            },
            py::return_value_policy::reference_internal)
        .def("vtkCellGhostTypes", []( const Mesh& mesh ) -> typename Mesh::VTKTypesArrayType const& {
                return mesh.vtkCellGhostTypes();
            },
            py::return_value_policy::reference_internal)
    ;
}
