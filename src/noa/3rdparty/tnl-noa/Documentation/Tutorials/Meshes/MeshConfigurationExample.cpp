#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag {};

//! [Configuration example]
namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// Create a template specialization of the tag specifying the MeshConfig template to use as the Config parameter for the mesh.
template<>
struct MeshConfigTemplateTag< MyConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = short int >
   struct MeshConfig
      : public DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return subentityDimension == 0 && entityDimension >= Cell::dimension - 1;
      }
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
//! [Configuration example]

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >
{ enum { enabled = false }; };

template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { enum { enabled = true }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

int main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = [] ( auto& reader, auto&& mesh ) -> bool
   {
      return true;
   };
   return ! TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
}
