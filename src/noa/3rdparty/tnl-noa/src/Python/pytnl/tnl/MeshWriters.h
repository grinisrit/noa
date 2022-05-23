#include "../iostream_caster.h"
#include <TNL/Meshes/VTKTraits.h>

// helper struct is needed to ensure correct initialization order in the PyWriter constructor
struct PyOstreamHelper
{
   py::object obj;
   pystreambuf::ostream str;

   PyOstreamHelper( py::object src )
      : obj(py::reinterpret_borrow<py::object>(src)),
        str(obj)
   {}
};

template< typename Writer, TNL::Meshes::VTK::FileFormat default_format >
struct PyWriter : public PyOstreamHelper, public Writer
{
   PyWriter( py::object src, TNL::Meshes::VTK::FileFormat format = default_format )
   : PyOstreamHelper(src), Writer(str)
   {}
};
