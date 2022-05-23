#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/Assert.h>

struct NotImplementedError
   : public std::runtime_error
{
   NotImplementedError( const char* msg )
   : std::runtime_error( msg )
   {}
};

static void register_exceptions( py::module & m )
{
    py::register_exception_translator(
        [](std::exception_ptr p) {
            try {
                if (p) std::rethrow_exception(p);
            }
            // translate exceptions used in the bindings
            catch (const NotImplementedError & e) {
                PyErr_SetString(PyExc_NotImplementedError, e.what());
            }
            // translate TNL::Assert::AssertionError
            catch (const TNL::Assert::AssertionError & e) {
                PyErr_SetString(PyExc_AssertionError, e.what());
            }
        }
    );
}
