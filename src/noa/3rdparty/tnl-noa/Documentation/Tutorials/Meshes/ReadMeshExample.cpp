//! [config]
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle > { enum { enabled = true }; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL
//! [config]

//! [task]
// Define the main task/function of the program
template< typename Mesh >
bool task( const Mesh& mesh, const std::string& inputFileName )
{
   std::cout << "The file '" << inputFileName << "' contains the following mesh: "
             << TNL::getType<Mesh>() << std::endl;
   return true;
}
//! [task]

//! [main]
int main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return task( mesh, inputFileName );
   };
   return ! TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
}
//! [main]
