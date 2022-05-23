#include <random>

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>

using namespace TNL;

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >
{ enum { enabled = false }; };

// Meshes are enabled only for topologies explicitly listed below.
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { enum { enabled = true }; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle > { enum { enabled = true }; };
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { enum { enabled = true }; };
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron > { enum { enabled = true }; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MyConfigTag, CellTopology, SpaceDimension >
{ enum { enabled = ( SpaceDimension == CellTopology::dimension ) }; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MyConfigTag, float > { enum { enabled = false }; };
template<> struct MeshRealTag< MyConfigTag, double > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MyConfigTag, int > { enum { enabled = true }; };
template<> struct MeshGlobalIndexTag< MyConfigTag, long int > { enum { enabled = false }; };
template<> struct MeshLocalIndexTag< MyConfigTag, short int > { enum { enabled = true }; };

// Config tag specifying the MeshConfig template to use.
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
      static constexpr bool subentityStorage( int entityDimension, int SubentityDimension )
      {
         return SubentityDimension == 0 && entityDimension >= Cell::dimension - 1;
      }

      static constexpr bool superentityStorage( int entityDimension, int SuperentityDimension )
      {
//         return false;
         return (entityDimension == 0 || entityDimension == Cell::dimension - 1) && SuperentityDimension == Cell::dimension;
      }

      static constexpr bool entityTagsStorage( int entityDimension )
      {
//         return false;
         return entityDimension == 0 || entityDimension >= Cell::dimension - 1;
      }

      static constexpr bool dualGraphStorage()
      {
         return true;
      }

      static constexpr int dualGraphMinCommonVertices = 1;
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


template< typename Mesh >
bool runGameOfLife( const Mesh& mesh )
{
   //! [Data vectors]
   using Index = typename Mesh::GlobalIndexType;
   const Index cellsCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() >();

   using VectorType = Containers::Vector< std::uint8_t, typename Mesh::DeviceType, Index >;
   VectorType f_in( cellsCount ), f_out( cellsCount );
   f_in.setValue( 0 );
   //! [Data vectors]

#if 1
   // generate random initial condition
   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_int_distribution<> dist(0, 1);
   for( Index i = 0; i < cellsCount; i++ )
      f_in[ i ] = dist(rng);
   const Index max_iter = 100;
#else
   // find a reference cell - the one closest to the point
   typename Mesh::PointType p = {0.5, 0.5};
   typename Mesh::RealType dist = 1e5;
   Index reference_cell = 0;
   for( Index i = 0; i < cellsCount; i++ ) {
      const auto cell = mesh.template getEntity< Mesh::getMeshDimension() >( i );
      const auto c = getEntityCenter( mesh, cell );
      const auto d = TNL::l2Norm( c - p );
      if( d < dist ) {
         reference_cell = i;
         dist = d;
      }
   }
   // R-pentomino (stabilizes after 1103 iterations)
   const Index max_iter = 1103;
   f_in[reference_cell] = 1;
   Index n1 = mesh.getCellNeighborIndex(reference_cell,1);  // bottom
   Index n2 = mesh.getCellNeighborIndex(reference_cell,2);  // left
   Index n3 = mesh.getCellNeighborIndex(reference_cell,5);  // top
   Index n4 = mesh.getCellNeighborIndex(reference_cell,6);  // top-right
   f_in[n1] = 1;
   f_in[n2] = 1;
   f_in[n3] = 1;
   f_in[n4] = 1;
/*
   // Acorn (stabilizes after 5206 iterations)
   const Index max_iter = 5206;
   f_in[reference_cell] = 1;
   Index n1 = mesh.getCellNeighborIndex(reference_cell,4);
   f_in[n1] = 1;
   Index s1 = mesh.getCellNeighborIndex(n1,4);
   Index s2 = mesh.getCellNeighborIndex(s1,4);
   Index n2 = mesh.getCellNeighborIndex(s2,4);
   f_in[n2] = 1;
   Index n3 = mesh.getCellNeighborIndex(n2,4);
   f_in[n3] = 1;
   Index n4 = mesh.getCellNeighborIndex(n3,4);
   f_in[n4] = 1;
   f_in[mesh.getCellNeighborIndex(s2,5)] = 1;
   f_in[mesh.getCellNeighborIndex(mesh.getCellNeighborIndex(n1,5),5)] = 1;
*/
#endif

   //! [make_snapshot]
   auto make_snapshot = [&] ( Index iteration )
   {
      const std::string filePath = "GoL." + std::to_string(iteration) + ".vtu";
      std::ofstream file( filePath );
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      Writer writer( file );
      writer.writeMetadata( iteration, iteration );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      writer.writeCellData( f_in, "function values" );
   };
   //! [make_snapshot]

   //! [make initial snapshot]
   // write initial state
   make_snapshot( 0 );
   //! [make initial snapshot]

   // captures for the iteration kernel
   auto f_in_view = f_in.getConstView();
   auto f_out_view = f_out.getView();
   Pointers::DevicePointer< const Mesh > meshDevicePointer( mesh );
   const Mesh* meshPointer = &meshDevicePointer.template getData< typename Mesh::DeviceType >();

   bool all_done = false;
   Index iteration = 0;
   do {
      iteration++;
      std::cout << "Computing iteration " << iteration << "..." << std::endl;

      //! [Game of Life kernel]
      auto kernel = [f_in_view, f_out_view, meshPointer] __cuda_callable__ ( Index i ) mutable
      {
         // sum values of the function on the neighbor cells
         typename VectorType::RealType sum = 0;
         for( Index n = 0; n < meshPointer->getCellNeighborsCount( i ); n++ ) {
            const Index neighbor = meshPointer->getCellNeighborIndex( i, n );
            sum += f_in_view[ neighbor ];
         }
         const bool live = f_in_view[ i ];

         // Conway's rules for square grid
         if( live ) {
            // any live cell with less than two live neighbors dies
            if( sum < 2 )
               f_out_view[ i ] = 0;
            // any live cell with two or three live neighbors survives
            else if( sum < 4 )
               f_out_view[ i ] = 1;
            // any live cell with more than three live neighbors dies
            else
               f_out_view[ i ] = 0;
         }
         else {
            // any dead cell with exactly three live neighbors becomes a live cell
            if( sum == 3 )
               f_out_view[ i ] = 1;
            // any other dead cell remains dead
            else
               f_out_view[ i ] = 0;
         }
      };
      //! [Game of Life kernel]

      //! [Game of Life iteration]
      // iterate over all cells
      mesh.template forAll< Mesh::getMeshDimension() >( kernel );

      // swap input and output arrays
      f_in.swap( f_out );
      // remember to update the views!
      f_in_view.bind( f_in.getView() );
      f_out_view.bind( f_out.getView() );

      // write output
      make_snapshot( iteration );

      // check if finished
      all_done = max( f_in ) == 0 || iteration > max_iter || f_in == f_out;
      //! [Game of Life iteration]
   }
   while( all_done == false );

   return true;
}

int main( int argc, char* argv[] )
{
   const std::string inputFileName = "grid-100x100.vtu";
   const std::string inputFileFormat = "auto";

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      return runGameOfLife( std::forward<MeshType>(mesh) );
   };
   return ! Meshes::resolveAndLoadMesh< MyConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
