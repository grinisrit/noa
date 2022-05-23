#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag {};

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

// Define the main task/function of the program
template< typename Mesh >
bool task( const Mesh& mesh )
{
   //! [getEntitiesCount]
   const int num_vertices = mesh.template getEntitiesCount< 0 >();
   const int num_cells = mesh.template getEntitiesCount< Mesh::getMeshDimension() >();
   //! [getEntitiesCount]

   // shut up warnings about unused variables
   (void) num_vertices;
   (void) num_cells;

   const int idx = num_vertices - 1;
   const int idx2 = num_cells - 1;

   //! [getEntity]
   typename Mesh::Vertex vert = mesh.template getEntity< 0 >( idx );
   typename Mesh::Cell elem = mesh.template getEntity< Mesh::getMeshDimension() >( idx2 );
   //! [getEntity]

   // shut up warnings about unused variables
   (void) vert;
   (void) elem;

{
   //! [Iteration over subentities]
   typename Mesh::Cell elem = mesh.template getEntity< Mesh::getMeshDimension() >( idx2 );
   const int n_subvert = elem.template getSubentitiesCount< 0 >();
   for( int v = 0; v < n_subvert; v++ ) {
      const int v_idx = elem.template getSubentityIndex< 0 >( v );
      typename Mesh::Vertex vert = mesh.template getEntity< 0 >( v_idx );
      // [Do some work...]
      (void) vert;
   }
   //! [Iteration over subentities]
}

{
   //! [Parallel iteration host]
   auto kernel = [&mesh] ( typename Mesh::GlobalIndexType i ) mutable
   {
      typename Mesh::Cell elem = mesh.template getEntity< Mesh::getMeshDimension() >( i );
      // [Do some work with the current cell `elem`...]
      (void) elem;
   };
   mesh.template forAll< Mesh::getMeshDimension() >( kernel );
   //! [Parallel iteration host]
}

   return true;
}

int main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = [] ( auto& reader, auto&& mesh ) -> bool
   {
      return task( mesh );
   };
   return ! TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
}
