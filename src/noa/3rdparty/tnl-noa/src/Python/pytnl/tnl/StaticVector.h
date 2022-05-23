#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "../tnl_indexing.h"

template< typename VectorType, typename Scope >
void export_StaticVector( Scope & scope, const char* name )
{
    using RealType = typename VectorType::RealType;

    auto vector = py::class_<VectorType>(scope, name)
        .def(py::init< RealType >())
        .def(py::init< VectorType >())
        .def("getSize", &VectorType::getSize)
        // operator=
        .def("assign", []( VectorType& vector, const VectorType& other ) -> VectorType& {
                return vector = other;
            })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("setValue", &VectorType::setValue)
        // TODO: pybind11
        // explicit namespace resolution is necessary, see http://stackoverflow.com/a/3084341/4180822
//        .def(py::self_ns::str(py::self))
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(py::self += RealType())
        .def(py::self -= RealType())
        .def(py::self *= RealType())
        .def(py::self /= RealType())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + RealType())
        .def(py::self - RealType())
        .def(py::self * RealType())
        .def(py::self / RealType())
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
    ;

    tnl_indexing< VectorType >( vector );
}
