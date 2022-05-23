#include <random>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/PVTUWriter.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

using namespace TNL;

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >
{ static constexpr bool enabled = false; };

// Meshes are enabled only for topologies explicitly listed below.
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MyConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MyConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< MyConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MyConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MyConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = GlobalIndex >
   struct MeshConfig
      : public DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return subentityDimension == 0 && entityDimension >= Cell::dimension - 1;
      }

      static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
      {
//         return false;
         return (entityDimension == 0 || entityDimension == Cell::dimension - 1) && superentityDimension == Cell::dimension;
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
   using LocalMesh = typename Mesh::MeshType;
   using Index = typename Mesh::GlobalIndexType;

   const LocalMesh& localMesh = mesh.getLocalMesh();

   // print basic mesh info
   mesh.printInfo( std::cout );

   // initialize the synchronizer
   using Synchronizer = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Mesh >;
   Synchronizer sync;
   sync.initialize( mesh );

   const Index pointsCount = localMesh.template getEntitiesCount< 0 >();
   const Index cellsCount = localMesh.template getEntitiesCount< Mesh::getMeshDimension() >();

   using VectorType = Containers::Vector< std::uint8_t, typename LocalMesh::DeviceType, Index >;
   VectorType f_in( cellsCount ), f_out( cellsCount );
   f_in.setValue( 0 );

/*
   // random initial condition
   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_int_distribution<> dist(0, 1);
   for( Index i = 0; i < cellsCount; i++ )
      f_in[ i ] = dist(rng);
   sync.synchronize( f_in );
*/
   // find the rank which contains most points in the box between (0.45, 0.45) and (0.55, 0.55)
   typename LocalMesh::PointType c1 = {0.48, 0.42};
   typename LocalMesh::PointType c2 = {0.58, 0.52};
   Index count = 0;
   for( Index i = 0; i < pointsCount; i++ ) {
      auto p = localMesh.getPoint(i);
      if( p.x() >= c1.x() && p.y() >= c1.y() && p.x() <= c2.x() && p.y() <= c2.y() ) {
         count++;
      }
   }
   Index max_count;
   TNL::MPI::Allreduce( &count, &max_count, 1, MPI_MAX, mesh.getCommunicator() );
   std::cout << "Rank " << TNL::MPI::GetRank() << ": count=" << count << ", max_count=" << max_count << std::endl;
   // FIXME: this is not reliable
   Index reference_cell = 0;
   if( count == max_count ) {
      // find cell which has all points in the central box
      for( Index i = 0; i < cellsCount; i++ ) {
         const Index subvertices = localMesh.template getSubentitiesCount< LocalMesh::getMeshDimension(), 0 >( i );
         int in_box = 0;
         for( Index j = 0; j < subvertices; j++ ) {
            auto p = localMesh.getPoint( localMesh.template getSubentityIndex< LocalMesh::getMeshDimension(), 0 >( i, j ) );
            if( p.x() >= c1.x() && p.y() >= c1.y() && p.x() <= c2.x() && p.y() <= c2.y() )
               in_box++;
         }
         if( in_box == subvertices ) {
            reference_cell = i;
         }
      }
   }
   // R-pentomino (stabilizes after 1103 iterations)
   const Index max_iter = 1103;
   if( count == max_count && localMesh.getCellNeighborsCount(reference_cell) > 6 ) {
      f_in[reference_cell] = 1;
      Index n1 = localMesh.getCellNeighborIndex(reference_cell,1);  // bottom
      Index n2 = localMesh.getCellNeighborIndex(reference_cell,2);  // left
      Index n3 = localMesh.getCellNeighborIndex(reference_cell,5);  // top
      Index n4 = localMesh.getCellNeighborIndex(reference_cell,6);  // top-right
      f_in[n1] = 1;
      f_in[n2] = 1;
      f_in[n3] = 1;
      f_in[n4] = 1;
   }
/*
   // Acorn (stabilizes after 5206 iterations)
   const Index max_iter = 5206;
   if( count == max_count ) {
      f_in[reference_cell] = 1;
      Index n1 = localMesh.getCellNeighborIndex(reference_cell,4);
      f_in[n1] = 1;
      Index s1 = localMesh.getCellNeighborIndex(n1,4);
      Index s2 = localMesh.getCellNeighborIndex(s1,4);
      Index n2 = localMesh.getCellNeighborIndex(s2,4);
      f_in[n2] = 1;
      Index n3 = localMesh.getCellNeighborIndex(n2,4);
      f_in[n3] = 1;
      Index n4 = localMesh.getCellNeighborIndex(n3,4);
      f_in[n4] = 1;
      f_in[localMesh.getCellNeighborIndex(s2,5)] = 1;
      f_in[localMesh.getCellNeighborIndex(localMesh.getCellNeighborIndex(n1,5),5)] = 1;
   }
*/

   auto make_snapshot = [&] ( Index iteration )
   {
      // create a .pvtu file (only rank 0 actually writes to the file)
      const std::string mainFilePath = "GoL." + std::to_string(iteration) + ".pvtu";
      std::ofstream file;
      if( TNL::MPI::GetRank() == 0 )
         file.open( mainFilePath );
      using PVTU = Meshes::Writers::PVTUWriter< LocalMesh >;
      PVTU pvtu( file );
      pvtu.template writeEntities< Mesh::getMeshDimension() >( mesh );
      pvtu.writeMetadata( iteration, iteration );
      // the PointData and CellData from the individual files should be added here
      if( mesh.getGhostLevels() > 0 )
         pvtu.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
      pvtu.template writePCellData< typename VectorType::RealType >( "function values" );
      const std::string subfilePath = pvtu.addPiece( mainFilePath, mesh.getCommunicator() );

      // create a .vtu file for local data
      std::ofstream subfile( subfilePath );
      using Writer = Meshes::Writers::VTUWriter< LocalMesh >;
      Writer writer( subfile );
      writer.writeMetadata( iteration, iteration );
      writer.template writeEntities< LocalMesh::getMeshDimension() >( localMesh );
      if( mesh.getGhostLevels() > 0 )
         writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
      writer.writeCellData( f_in, "function values" );
   };

   // write initial state
   make_snapshot( 0 );

   // captures for the iteration kernel
   auto f_in_view = f_in.getConstView();
   auto f_out_view = f_out.getView();
   Pointers::DevicePointer< const LocalMesh > localMeshDevicePointer( localMesh );
   const LocalMesh* localMeshPointer = &localMeshDevicePointer.template getData< typename LocalMesh::DeviceType >();

   bool all_done = false;
   Index iteration = 0;
   do {
      iteration++;
      if( TNL::MPI::GetRank() == 0 )
         std::cout << "Computing iteration " << iteration << "..." << std::endl;

      // iterate over all local cells
      auto kernel = [f_in_view, f_out_view, localMeshPointer] __cuda_callable__ ( Index i ) mutable
      {
         // sum values of the function on the neighbor cells
         typename VectorType::RealType sum = 0;
         for( Index n = 0; n < localMeshPointer->getCellNeighborsCount( i ); n++ ) {
            const Index neighbor = localMeshPointer->getCellNeighborIndex( i, n );
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
      localMesh.template forLocal< Mesh::getMeshDimension() >( kernel );

      // synchronize
      sync.synchronize( f_out );

      // swap input and output arrays
      f_in.swap( f_out );
      // remember to update the views!
      f_in_view.bind( f_in.getView() );
      f_out_view.bind( f_out.getView() );

      // write output
      make_snapshot( iteration );

      // check if finished
      const bool done = max( f_in ) == 0 || iteration > max_iter || f_in == f_out;
      TNL::MPI::Allreduce( &done, &all_done, 1, MPI_LAND, mesh.getCommunicator() );
   }
   while( all_done == false );

   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addEntry< String >( "input-file-format", "Input mesh file format.", "auto" );
   config.addDelimiter( "MPI settings:" );
   TNL::MPI::configSetup( config );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   TNL::MPI::ScopedInitializer mpi(argc, argv);

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String inputFileFormat = parameters.getParameter< String >( "input-file-format" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      return runGameOfLife( std::forward<MeshType>(mesh) );
   };
   return ! Meshes::resolveAndLoadDistributedMesh< MyConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
