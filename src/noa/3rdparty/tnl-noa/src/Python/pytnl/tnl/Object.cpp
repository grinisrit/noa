#include <pybind11/pybind11.h>
namespace py = pybind11;

// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include <TNL/Object.h>

void export_Object( py::module & m )
{
    py::class_< TNL::Object >( m, "Object" )
        // TODO: make it abstract class in Python
        .def("save", (void (TNL::Object::*)(const TNL::String &) const) &TNL::Object::save)
        .def("load", (void (TNL::Object::*)(const TNL::String &)) &TNL::Object::load)
        // FIXME: why does it not work?
//        .def("save", py::overload_cast<TNL::File>(&TNL::Object::save, py::const_))
//        .def("load", py::overload_cast<TNL::File>(&TNL::Object::load))
        .def("save", (void (TNL::Object::*)(TNL::File &) const) &TNL::Object::save)
        .def("load", (void (TNL::Object::*)(TNL::File &)) &TNL::Object::load)
    ;
}
