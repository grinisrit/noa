#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Readers/VTUReader.h>
#include <TNL/Meshes/Readers/VTIReader.h>

// trampoline classes needed for overriding virtual methods
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html

class PyMeshReader
: public TNL::Meshes::Readers::MeshReader
{
   using Parent = TNL::Meshes::Readers::MeshReader;

public:
   // inherit constructors
   using TNL::Meshes::Readers::MeshReader::MeshReader;

   // trampolines (one for each virtual method)
   void reset() override
   {
      PYBIND11_OVERRIDE_PURE( void, Parent, reset );
   }

   void detectMesh() override
   {
      PYBIND11_OVERRIDE_PURE( void, Parent, detectMesh );
   }
};

class PyXMLVTK
: public TNL::Meshes::Readers::XMLVTK
{
   using Parent = TNL::Meshes::Readers::XMLVTK;

public:
   // inherit constructors
   using TNL::Meshes::Readers::XMLVTK::XMLVTK;

   // trampolines (one for each virtual method)
   void reset() override
   {
      PYBIND11_OVERRIDE_PURE( void, Parent, reset );
   }

   void detectMesh() override
   {
      PYBIND11_OVERRIDE_PURE( void, Parent, detectMesh );
   }
};
