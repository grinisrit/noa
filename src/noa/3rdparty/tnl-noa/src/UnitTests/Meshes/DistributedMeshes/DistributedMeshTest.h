#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <unordered_map>
#include <filesystem>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/VTKTraits.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/Writers/PVTUWriter.h>
#include <TNL/Meshes/Readers/PVTUReader.h>

namespace DistributedMeshTest {

namespace fs = std::filesystem;

using namespace TNL;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;

// cannot be deduced from the grid
using LocalIndexType = short int;

template< typename Mesh >
struct GridDistributor;

template< typename Real, typename Device, typename Index >
struct GridDistributor< TNL::Meshes::Grid< 2, Real, Device, Index > >
{
   using GridType = TNL::Meshes::Grid< 2, Real, Device, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using CellTopology = TNL::Meshes::Topologies::Quadrangle;
   using MeshConfig = TNL::Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using LocalMeshType = TNL::Meshes::Mesh< MeshConfig >;

   GridDistributor() = delete;

   GridDistributor( CoordinatesType rank_sizes, MPI_Comm communicator )
      : rank(TNL::MPI::GetRank(communicator)),
        nproc(TNL::MPI::GetSize(communicator)),
        rank_sizes(rank_sizes),
        communicator(communicator)
   {}

   void decompose( const GridType& grid,
                   DistributedMesh< LocalMeshType >& mesh,
                   int overlap )
   {
      ASSERT_EQ( nproc, product(rank_sizes) );

      // subdomain coordinates
      const CoordinatesType rank_coordinates = {rank % rank_sizes.x(), rank / rank_sizes.x()};

      // local mesh
      local_size = grid.getDimensions() / rank_sizes;
      ASSERT_EQ( local_size * rank_sizes, grid.getDimensions() );
      // ranges for local (owned) cells
      cell_begin = rank_coordinates * local_size;
      cell_end = (rank_coordinates + 1) * local_size;
      ASSERT_LE( cell_begin, grid.getDimensions() );
      ASSERT_LE( cell_end, grid.getDimensions() );
      // ranges for local (owned) vertices
      vert_begin = cell_begin + 1;
      if( rank_coordinates.x() == 0 ) vert_begin.x()--;
      if( rank_coordinates.y() == 0 ) vert_begin.y()--;
      vert_end = cell_end + 1;

      // count local (owned) entities
      localVerticesCount = product(vert_end - vert_begin);
      localCellsCount = product(cell_end - cell_begin);
      // set entities counts on the subdomain
      verticesCount = 0;
      for( Index y = vert_begin.y() - (rank_coordinates.y() > 0) * (overlap + (overlap>0)); y < vert_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = vert_begin.x() - (rank_coordinates.x() > 0) * (overlap + (overlap>0)); x < vert_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         verticesCount++;
      cellsCount = 0;
      for( Index y = cell_begin.y() - (rank_coordinates.y() > 0) * overlap; y < cell_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = cell_begin.x() - (rank_coordinates.x() > 0) * overlap; x < cell_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         cellsCount++;
      TNL::Meshes::MeshBuilder< LocalMeshType > meshBuilder;
      meshBuilder.setEntitiesCount( verticesCount, cellsCount );

      // mappings for vertex indices
      std::unordered_map< Index, Index > vert_global_to_local, cell_global_to_local;

      Index idx = 0;
      auto add_vertex = [&] ( Index x, Index y )
      {
         if( x < 0 || x > grid.getDimensions().x() ) return;
         if( y < 0 || y > grid.getDimensions().y() ) return;
         typename GridType::Vertex vertex( grid );
         vertex.setCoordinates( {x, y} );
         vertex.refresh();
         if( vert_global_to_local.count( vertex.getIndex() ) == 0 ) {
            meshBuilder.setPoint( idx, vertex.getCenter() );
            vert_global_to_local.insert( {vertex.getIndex(), idx} );
            idx++;
         }
      };
      auto add_cell = [&] ( Index x, Index y )
      {
         if( x < 0 || x >= grid.getDimensions().x() ) return;
         if( y < 0 || y >= grid.getDimensions().y() ) return;
         typename GridType::Cell cell( grid );
         cell.setCoordinates( {x, y} );
         cell.refresh();
         if( cell_global_to_local.count( cell.getIndex() ) == 0 ) {
            meshBuilder.getCellSeed( idx ).setCornerId( 0, vert_global_to_local[ cell.template getNeighbourEntity< 0 >( { 0, 0 } ).getIndex() ] );
            meshBuilder.getCellSeed( idx ).setCornerId( 1, vert_global_to_local[ cell.template getNeighbourEntity< 0 >( { 1, 0 } ).getIndex() ] );
            meshBuilder.getCellSeed( idx ).setCornerId( 2, vert_global_to_local[ cell.template getNeighbourEntity< 0 >( { 1, 1 } ).getIndex() ] );
            meshBuilder.getCellSeed( idx ).setCornerId( 3, vert_global_to_local[ cell.template getNeighbourEntity< 0 >( { 0, 1 } ).getIndex() ] );
            cell_global_to_local.insert( {cell.getIndex(), idx} );
            idx++;
         }
      };

      // add all local (owned) vertices
      for( Index y = vert_begin.y(); y < vert_end.y(); y++ )
      for( Index x = vert_begin.x(); x < vert_end.x(); x++ )
         add_vertex( x, y );
      // add remaining vertices that will be needed for overlaps
//      for( Index y = vert_begin.y() - (rank_coordinates.y() > 0) * (overlap + (overlap>0)); y < vert_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
//      for( Index x = vert_begin.x() - (rank_coordinates.x() > 0) * (overlap + (overlap>0)); x < vert_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
//         add_vertex( x, y );
      // - bottom left
      for( Index y = vert_begin.y() - (rank_coordinates.y() > 0) * (overlap + (overlap>0)); y < vert_begin.y(); y++ )
      for( Index x = vert_begin.x() - (rank_coordinates.x() > 0) * (overlap + (overlap>0)); x < vert_begin.x(); x++ )
         add_vertex( x, y );
      // - bottom
      for( Index y = vert_begin.y() - (rank_coordinates.y() > 0) * (overlap + (overlap>0)); y < vert_begin.y(); y++ )
      for( Index x = vert_begin.x(); x < vert_end.x(); x++ )
         add_vertex( x, y );
      // - bottom right
      for( Index y = vert_begin.y() - (rank_coordinates.y() > 0) * (overlap + (overlap>0)); y < vert_begin.y(); y++ )
      for( Index x = vert_end.x(); x < vert_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_vertex( x, y );
      // - left
      for( Index y = vert_begin.y(); y < vert_end.y(); y++ )
      for( Index x = vert_begin.x() - (rank_coordinates.x() > 0) * (overlap + (overlap>0)); x < vert_begin.x(); x++ )
         add_vertex( x, y );
      // - right
      for( Index y = vert_begin.y(); y < vert_end.y(); y++ )
      for( Index x = vert_end.x(); x < vert_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_vertex( x, y );
      // - top left
      for( Index y = vert_end.y(); y < vert_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = vert_begin.x() - (rank_coordinates.x() > 0) * (overlap + (overlap>0)); x < vert_begin.x(); x++ )
         add_vertex( x, y );
      // - top
      for( Index y = vert_end.y(); y < vert_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = vert_begin.x(); x < vert_end.x(); x++ )
         add_vertex( x, y );
      // - top right
      for( Index y = vert_end.y(); y < vert_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = vert_end.x(); x < vert_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_vertex( x, y );

      // reset counter
      idx = 0;

      // add local cells
      for( Index y = cell_begin.y(); y < cell_end.y(); y++ )
      for( Index x = cell_begin.x(); x < cell_end.x(); x++ )
         add_cell( x, y );
      // add ghost cells
//      for( Index y = cell_begin.y() - (rank_coordinates.y() > 0) * overlap; y < cell_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
//      for( Index x = cell_begin.x() - (rank_coordinates.x() > 0) * overlap; x < cell_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
//         add_cell( x, y );
      // - bottom left
      for( Index y = cell_begin.y() - (rank_coordinates.y() > 0) * overlap; y < cell_begin.y(); y++ )
      for( Index x = cell_begin.x() - (rank_coordinates.x() > 0) * overlap; x < cell_begin.x(); x++ )
         add_cell( x, y );
      // - bottom
      for( Index y = cell_begin.y() - (rank_coordinates.y() > 0) * overlap; y < cell_begin.y(); y++ )
      for( Index x = cell_begin.x(); x < cell_end.x(); x++ )
         add_cell( x, y );
      // - bottom right
      for( Index y = cell_begin.y() - (rank_coordinates.y() > 0) * overlap; y < cell_begin.y(); y++ )
      for( Index x = cell_end.x(); x < cell_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_cell( x, y );
      // - left
      for( Index y = cell_begin.y(); y < cell_end.y(); y++ )
      for( Index x = cell_begin.x() - (rank_coordinates.x() > 0) * overlap; x < cell_begin.x(); x++ )
         add_cell( x, y );
      // - right
      for( Index y = cell_begin.y(); y < cell_end.y(); y++ )
      for( Index x = cell_end.x(); x < cell_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_cell( x, y );
      // - top left
      for( Index y = cell_end.y(); y < cell_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = cell_begin.x() - (rank_coordinates.x() > 0) * overlap; x < cell_begin.x(); x++ )
         add_cell( x, y );
      // - top
      for( Index y = cell_end.y(); y < cell_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = cell_begin.x(); x < cell_end.x(); x++ )
         add_cell( x, y );
      // - top right
      for( Index y = cell_end.y(); y < cell_end.y() + (rank_coordinates.y() < rank_sizes.y() - 1) * overlap; y++ )
      for( Index x = cell_end.x(); x < cell_end.x() + (rank_coordinates.x() < rank_sizes.x() - 1) * overlap; x++ )
         add_cell( x, y );

      ASSERT_TRUE( meshBuilder.build( mesh.getLocalMesh() ) );

      // set ghost levels
      mesh.setGhostLevels( overlap );

      if( overlap > 0 ) {
         // assign point ghost tags
         mesh.vtkPointGhostTypes().setSize( verticesCount );
         for( Index i = 0; i < localVerticesCount; i++ )
            mesh.vtkPointGhostTypes()[ i ] = 0;
         for( Index i = localVerticesCount; i < verticesCount; i++ ) {
            mesh.vtkPointGhostTypes()[ i ] = (std::uint8_t) VTK::PointGhostTypes::DUPLICATEPOINT;
            mesh.getLocalMesh().template addEntityTag< 0 >( i, EntityTags::GhostEntity );
         }

         // assign cell ghost tags
         mesh.vtkCellGhostTypes().setSize( cellsCount );
         for( Index i = 0; i < localCellsCount; i++ )
            mesh.vtkCellGhostTypes()[ i ] = 0;
         for( Index i = localCellsCount; i < cellsCount; i++ ) {
            mesh.vtkCellGhostTypes()[ i ] = (std::uint8_t) VTK::CellGhostTypes::DUPLICATECELL;
            mesh.getLocalMesh().template addEntityTag< 2 >( i, EntityTags::GhostEntity );
         }

         // update the entity tags layers after setting ghost indices
         mesh.getLocalMesh().template updateEntityTagsLayer< 0 >();
         mesh.getLocalMesh().template updateEntityTagsLayer< 2 >();

         // assign global indices
         auto& points_indices = mesh.template getGlobalIndices< 0 >();
         auto& cells_indices = mesh.template getGlobalIndices< 2 >();
         points_indices.setSize( verticesCount );
         cells_indices.setSize( cellsCount );
         // compute mapping between old and new global indices (global indices on the distributed mesh != global indices on the original grid)
         const auto vert_new_global_indices = renumberVertices( grid, rank_sizes );
         const auto cell_new_global_indices = renumberCells( grid, rank_sizes );
         for( auto pair : vert_global_to_local )
            points_indices[ pair.second ] = vert_new_global_indices.at( pair.first );
         for( auto pair : cell_global_to_local )
            cells_indices[ pair.second ] = cell_new_global_indices.at( pair.first );
      }

      // set the communicator
      mesh.setCommunicator( communicator );

      if( overlap > 0 ) {
         // distribute faces
         distributeSubentities< 1 >( mesh, /* preferHighRanks = */ false );
      }
   }

   static std::map< Index, Index > renumberVertices( const GridType& grid, CoordinatesType rank_sizes )
   {
      std::map< Index, Index > result;
      Index idx = 0;
      const CoordinatesType local_size = grid.getDimensions() / rank_sizes;
      CoordinatesType rank_coordinates;
      for( rank_coordinates.y() = 0; rank_coordinates.y() < rank_sizes.y(); rank_coordinates.y()++ )
      for( rank_coordinates.x() = 0; rank_coordinates.x() < rank_sizes.x(); rank_coordinates.x()++ )
      {
         const CoordinatesType cell_begin = rank_coordinates * local_size;
         const CoordinatesType cell_end = (rank_coordinates + 1) * local_size;
         CoordinatesType vert_begin = cell_begin + 1;
         if( rank_coordinates.x() == 0 ) vert_begin.x()--;
         if( rank_coordinates.y() == 0 ) vert_begin.y()--;
         const CoordinatesType vert_end = cell_end + 1;

         for( Index y = vert_begin.y(); y < vert_end.y(); y++ )
         for( Index x = vert_begin.x(); x < vert_end.x(); x++ )
         {
            typename GridType::Vertex vertex( grid );
            vertex.setCoordinates( {x, y} );
            vertex.refresh();
            result.insert( {vertex.getIndex(), idx} );
            idx++;
         }
      }
      return result;
   }

   static std::map< Index, Index > renumberCells( const GridType& grid, CoordinatesType rank_sizes )
   {
      Index idx = 0;
      std::map< Index, Index > result;
      const CoordinatesType local_size = grid.getDimensions() / rank_sizes;
      CoordinatesType rank_coordinates;
      for( rank_coordinates.y() = 0; rank_coordinates.y() < rank_sizes.y(); rank_coordinates.y()++ )
      for( rank_coordinates.x() = 0; rank_coordinates.x() < rank_sizes.x(); rank_coordinates.x()++ )
      {
         const CoordinatesType cell_begin = rank_coordinates * local_size;
         const CoordinatesType cell_end = (rank_coordinates + 1) * local_size;

         for( Index y = cell_begin.y(); y < cell_end.y(); y++ )
         for( Index x = cell_begin.x(); x < cell_end.x(); x++ )
         {
            typename GridType::Cell cell( grid );
            cell.setCoordinates( {x, y} );
            cell.refresh();
            result.insert( {cell.getIndex(), idx} );
            idx++;
         }
      }
      return result;
   }

   // input parameters
   int rank, nproc;
   CoordinatesType rank_sizes;
   MPI_Comm communicator;
   // output attributes (byproduct of the decomposition, useful for testing)
   CoordinatesType rank_coordinates, local_size, vert_begin, vert_end, cell_begin, cell_end;
   Index verticesCount, cellsCount, localVerticesCount, localCellsCount;
};

template< typename Mesh, typename Distributor >
void validateMesh( const Mesh& mesh, const Distributor& distributor, int ghostLevels )
{
   using Index = typename Mesh::GlobalIndexType;
   using Device = typename Mesh::DeviceType;

   // check basic interface
   EXPECT_EQ( mesh.getCommunicator(), MPI_COMM_WORLD );
   EXPECT_EQ( mesh.getGhostLevels(), ghostLevels );
   if( ghostLevels > 0 ) {
      EXPECT_EQ( mesh.template getGlobalIndices< 0 >().getSize(), mesh.getLocalMesh().template getEntitiesCount< 0 >() );
      EXPECT_EQ( mesh.template getGlobalIndices< 2 >().getSize(), mesh.getLocalMesh().template getEntitiesCount< 2 >() );
      EXPECT_EQ( mesh.vtkPointGhostTypes().getSize(), mesh.getLocalMesh().template getEntitiesCount< 0 >() );
      EXPECT_EQ( mesh.vtkCellGhostTypes().getSize(), mesh.getLocalMesh().template getEntitiesCount< 2 >() );
   }
   else {
      EXPECT_EQ( mesh.template getGlobalIndices< 0 >().getSize(), 0 );
      EXPECT_EQ( mesh.template getGlobalIndices< 2 >().getSize(), 0 );
      EXPECT_EQ( mesh.vtkPointGhostTypes().getSize(), 0 );
      EXPECT_EQ( mesh.vtkCellGhostTypes().getSize(), 0 );
   }

   // check entities counts
   EXPECT_EQ( mesh.getLocalMesh().template getEntitiesCount< 0 >(), distributor.verticesCount );
   EXPECT_EQ( mesh.getLocalMesh().template getEntitiesCount< 2 >(), distributor.cellsCount );
   EXPECT_EQ( mesh.getLocalMesh().template getGhostEntitiesCount< 0 >(), distributor.verticesCount - distributor.localVerticesCount );
   EXPECT_EQ( mesh.getLocalMesh().template getGhostEntitiesCount< 2 >(), distributor.cellsCount - distributor.localCellsCount );
   EXPECT_EQ( mesh.getLocalMesh().template getGhostEntitiesOffset< 0 >(), distributor.localVerticesCount );
   EXPECT_EQ( mesh.getLocalMesh().template getGhostEntitiesOffset< 2 >(), distributor.localCellsCount );

   if( ghostLevels > 0 ) {
      // check that vtkPointGhostTypes is consistent with the tags array
      for( Index i = 0; i < distributor.verticesCount; i++ ) {
         EXPECT_EQ( mesh.getLocalMesh().template isGhostEntity< 0 >( i ), mesh.vtkPointGhostTypes()[ i ] & (std::uint8_t) VTK::PointGhostTypes::DUPLICATEPOINT ) << "vertex idx = " << i;
      }
      // check that vtkPointGhostTypes and vtkCellGhostTypes are consistent with the entities tags arrays
      for( Index i = 0; i < distributor.cellsCount; i++ ) {
         EXPECT_EQ( mesh.getLocalMesh().template isGhostEntity< 2 >( i ), mesh.vtkCellGhostTypes()[ i ] & (std::uint8_t) VTK::CellGhostTypes::DUPLICATECELL ) << "cell idx = " << i;
      }
   }

   // check decomposition: check that ghost tags are set on correct entities
   for( Index i = 0; i < distributor.localVerticesCount; i++ ) {
      EXPECT_FALSE( mesh.getLocalMesh().template isGhostEntity< 0 >( i ) ) << "vertex idx = " << i;
   }
   if( ghostLevels > 0 )
   for( Index i = distributor.localVerticesCount; i < distributor.verticesCount; i++ ) {
      EXPECT_TRUE( mesh.getLocalMesh().template isGhostEntity< 0 >( i ) ) << "vertex idx = " << i;
   }
   for( Index i = 0; i < distributor.localCellsCount; i++ ) {
      EXPECT_FALSE( mesh.getLocalMesh().template isGhostEntity< 2 >( i ) ) << "cell idx = " << i;
   }
   if( ghostLevels > 0 )
   for( Index i = distributor.localCellsCount; i < distributor.cellsCount; i++ ) {
      EXPECT_TRUE( mesh.getLocalMesh().template isGhostEntity< 2 >( i ) ) << "cell idx = " << i;
   }

   if( ghostLevels > 0 ) {
      // exchange local vertices and cells counts, exclusive scan to compute offsets
      Containers::Vector< Index, Device > vert_offsets( distributor.nproc + 1 ), cell_offsets( distributor.nproc + 1 );
      {
         Containers::Array< Index, Device > vert_sendbuf( distributor.nproc ), cell_sendbuf( distributor.nproc );
         vert_sendbuf.setValue( distributor.localVerticesCount );
         cell_sendbuf.setValue( distributor.localCellsCount );
         TNL::MPI::Alltoall( vert_sendbuf.getData(), 1,
                             vert_offsets.getData(), 1,
                             distributor.communicator );
         TNL::MPI::Alltoall( cell_sendbuf.getData(), 1,
                             cell_offsets.getData(), 1,
                             distributor.communicator );
      }
      vert_offsets.setElement( distributor.nproc, 0 );
      cell_offsets.setElement( distributor.nproc, 0 );
      Algorithms::inplaceExclusiveScan( vert_offsets );
      Algorithms::inplaceExclusiveScan( cell_offsets );
      EXPECT_EQ( vert_offsets[ distributor.rank ], mesh.template getGlobalIndices< 0 >()[ 0 ] );
      EXPECT_EQ( cell_offsets[ distributor.rank ], mesh.template getGlobalIndices< 2 >()[ 0 ] );

      // check global indices of ghost entities
      for( Index i = 0; i < distributor.localVerticesCount; i++ ) {
         EXPECT_EQ( mesh.template getGlobalIndices< 0 >()[ i ], vert_offsets[ distributor.rank ] + i ) << "vertex idx = " << i;
      }
      for( Index i = distributor.localVerticesCount; i < distributor.verticesCount; i++ ) {
         EXPECT_TRUE( mesh.template getGlobalIndices< 0 >()[ i ] < vert_offsets[ distributor.rank ] ||
                      mesh.template getGlobalIndices< 0 >()[ i ] >= vert_offsets[ distributor.rank + 1 ] ) << "vertex idx = " << i;
      }
      for( Index i = 0; i < distributor.localCellsCount; i++ ) {
         EXPECT_EQ( mesh.template getGlobalIndices< 2 >()[ i ], cell_offsets[ distributor.rank ] + i ) << "cell idx = " << i;
      }
      for( Index i = distributor.localCellsCount; i < distributor.cellsCount; i++ ) {
         EXPECT_TRUE( mesh.template getGlobalIndices< 2 >()[ i ] < cell_offsets[ distributor.rank ] ||
                      mesh.template getGlobalIndices< 2 >()[ i ] >= cell_offsets[ distributor.rank + 1 ] ) << "cell idx = " << i;
      }
   }
}

struct TestEntitiesProcessor
{
   template< typename Mesh, typename UserData, typename Entity >
   __cuda_callable__
   static void processEntity( const Mesh& mesh, UserData& userData, const Entity& entity )
   {
      userData[ entity.getIndex() ] += 1;
   }
};

template< typename Device, typename EntityType, typename MeshType, typename HostArray >
void testIterationOnDevice( const MeshType& mesh,
                            const HostArray& expected_array_boundary,
                            const HostArray& expected_array_interior,
                            const HostArray& expected_array_all,
                            const HostArray& expected_array_ghost,
                            const HostArray& expected_array_local )
{
   using DeviceMesh = Mesh< typename MeshType::Config, Device >;
   Pointers::SharedPointer< DeviceMesh > meshPointer;
   *meshPointer = mesh;

   Containers::Array< int, Device > array_boundary( mesh.template getEntitiesCount< EntityType >() );
   Containers::Array< int, Device > array_interior( mesh.template getEntitiesCount< EntityType >() );
   Containers::Array< int, Device > array_all     ( mesh.template getEntitiesCount< EntityType >() );
   Containers::Array< int, Device > array_ghost   ( mesh.template getEntitiesCount< EntityType >() );
   Containers::Array< int, Device > array_local   ( mesh.template getEntitiesCount< EntityType >() );

   // test iteration methods: forAll, forBoundary, forInterior
   array_boundary.setValue( 0 );
   array_interior.setValue( 0 );
   array_all     .setValue( 0 );
   array_ghost   .setValue( 0 );
   array_local   .setValue( 0 );

   auto view_boundary = array_boundary.getView();
   auto view_interior = array_interior.getView();
   auto view_all      = array_all.getView();
   auto view_ghost    = array_ghost.getView();
   auto view_local    = array_local.getView();

   auto f_boundary = [view_boundary] __cuda_callable__ ( typename MeshType::GlobalIndexType i ) mutable { view_boundary[i] += 1; };
   auto f_interior = [view_interior] __cuda_callable__ ( typename MeshType::GlobalIndexType i ) mutable { view_interior[i] += 1; };
   auto f_all      = [view_all]      __cuda_callable__ ( typename MeshType::GlobalIndexType i ) mutable { view_all[i] += 1; };
   auto f_ghost    = [view_ghost]    __cuda_callable__ ( typename MeshType::GlobalIndexType i ) mutable { view_ghost[i] += 1; };
   auto f_local    = [view_local]    __cuda_callable__ ( typename MeshType::GlobalIndexType i ) mutable { view_local[i] += 1; };

   meshPointer->template forBoundary< EntityType::getEntityDimension() >( f_boundary );
   meshPointer->template forInterior< EntityType::getEntityDimension() >( f_interior );
   meshPointer->template forAll     < EntityType::getEntityDimension() >( f_all );
   meshPointer->template forGhost   < EntityType::getEntityDimension() >( f_ghost );
   meshPointer->template forLocal   < EntityType::getEntityDimension() >( f_local );

   EXPECT_EQ( array_boundary, expected_array_boundary );
   EXPECT_EQ( array_interior, expected_array_interior );
   EXPECT_EQ( array_all,      expected_array_all      );
   EXPECT_EQ( array_ghost,    expected_array_ghost    );
   EXPECT_EQ( array_local,    expected_array_local    );
}

template< typename Mesh >
void testIteration( const Mesh& mesh )
{
   const auto cellsCount = mesh.getLocalMesh().template getEntitiesCount< 2 >();
   const auto verticesCount = mesh.getLocalMesh().template getEntitiesCount< 0 >();

   // create arrays for all test cases
   Containers::Array< int > array_cells_boundary( cellsCount );
   Containers::Array< int > array_cells_interior( cellsCount );
   Containers::Array< int > array_cells_all     ( cellsCount );
   Containers::Array< int > array_cells_ghost   ( cellsCount );
   Containers::Array< int > array_cells_local   ( cellsCount );

   Containers::Array< int > array_vertices_boundary( verticesCount );
   Containers::Array< int > array_vertices_interior( verticesCount );
   Containers::Array< int > array_vertices_all     ( verticesCount );
   Containers::Array< int > array_vertices_ghost   ( verticesCount );
   Containers::Array< int > array_vertices_local   ( verticesCount );

   // set expected values
   for( int i = 0; i < cellsCount; i++ ) {
      array_cells_all[ i ] = 1;
      if( mesh.getLocalMesh().template isBoundaryEntity< 2 >( i ) ) {
         array_cells_boundary[ i ] = 1;
         array_cells_interior[ i ] = 0;
      }
      else {
         array_cells_boundary[ i ] = 0;
         array_cells_interior[ i ] = 1;
      }
      if( mesh.getLocalMesh().template isGhostEntity< 2 >( i ) ) {
         array_cells_ghost[ i ] = 1;
         array_cells_local[ i ] = 0;
      }
      else {
         array_cells_ghost[ i ] = 0;
         array_cells_local[ i ] = 1;
      }
   }
   for( int i = 0; i < verticesCount; i++ ) {
      array_vertices_all[ i ] = 1;
      if( mesh.getLocalMesh().template isBoundaryEntity< 0 >( i ) ) {
         array_vertices_boundary[ i ] = 1;
         array_vertices_interior[ i ] = 0;
      }
      else {
         array_vertices_boundary[ i ] = 0;
         array_vertices_interior[ i ] = 1;
      }
      if( mesh.getLocalMesh().template isGhostEntity< 0 >( i ) ) {
         array_vertices_ghost[ i ] = 1;
         array_vertices_local[ i ] = 0;
      }
      else {
         array_vertices_ghost[ i ] = 0;
         array_vertices_local[ i ] = 1;
      }
   }

   // test
   testIterationOnDevice< Devices::Host, typename Mesh::Cell >( mesh.getLocalMesh(), array_cells_boundary, array_cells_interior, array_cells_all, array_cells_ghost, array_cells_local );
   testIterationOnDevice< Devices::Host, typename Mesh::Vertex >( mesh.getLocalMesh(), array_vertices_boundary, array_vertices_interior, array_vertices_all, array_vertices_ghost, array_vertices_local );
#ifdef __CUDACC__
   testIterationOnDevice< Devices::Cuda, typename Mesh::Cell >( mesh.getLocalMesh(), array_cells_boundary, array_cells_interior, array_cells_all, array_cells_ghost, array_cells_local );
   testIterationOnDevice< Devices::Cuda, typename Mesh::Vertex >( mesh.getLocalMesh(), array_vertices_boundary, array_vertices_interior, array_vertices_all, array_vertices_ghost, array_vertices_local );
#endif
}

template< typename Device, typename EntityType, typename MeshType >
void testSynchronizerOnDevice_global_indices( const MeshType& mesh )
{
   using LocalMesh = Mesh< typename MeshType::Config, Device >;
   using DeviceMesh = DistributedMesh< LocalMesh >;
   using IndexType = typename MeshType::GlobalIndexType;
   using MeshFunction = Functions::MeshFunction< LocalMesh, EntityType::getEntityDimension(), IndexType >;
   using Synchronizer = DistributedMeshes::DistributedMeshSynchronizer< DeviceMesh, EntityType::getEntityDimension() >;

   // initialize
   DeviceMesh deviceMesh;
   deviceMesh = mesh;
   Pointers::SharedPointer< LocalMesh > localMeshPointer;
   *localMeshPointer = deviceMesh.getLocalMesh();
   MeshFunction f( localMeshPointer );
   f.getData().setValue( -1 );

   // set global indices of local entities
   for( IndexType i = 0; i < mesh.getLocalMesh().template getEntitiesCount< EntityType >(); i++ )
      if( ! mesh.getLocalMesh().template isGhostEntity< EntityType::getEntityDimension() >( i ) )
         f.getData().setElement( i, mesh.template getGlobalIndices< EntityType::getEntityDimension() >()[ i ] );

   // synchronize
   Synchronizer sync;
   sync.initialize( deviceMesh );
   sync.synchronize( f );

   // check all global indices
   EXPECT_EQ( f.getData(), mesh.template getGlobalIndices< EntityType::getEntityDimension() >() );
}

template< typename Mesh >
__cuda_callable__
typename Mesh::LocalIndexType
getCellsForFace( const Mesh & mesh, const typename Mesh::GlobalIndexType E, typename Mesh::GlobalIndexType* cellIndexes )
{
    using LocalIndexType = typename Mesh::LocalIndexType;
    const LocalIndexType numCells = mesh.template getSuperentitiesCount< Mesh::getMeshDimension() - 1, Mesh::getMeshDimension() >( E );
    for( LocalIndexType i = 0; i < numCells; i++ )
        cellIndexes[ i ] = mesh.template getSuperentityIndex< Mesh::getMeshDimension() - 1, Mesh::getMeshDimension() >( E, i );
    return numCells;
}

// testing global indices is not enough - entity centers are needed to ensure that the transferred data really match the physical entities
template< typename Device, typename EntityType, typename MeshType >
void testSynchronizerOnDevice_entity_centers( const MeshType& mesh )
{
   using LocalMesh = TNL::Meshes::Mesh< typename MeshType::Config, Device >;
   using DeviceMesh = TNL::Meshes::DistributedMeshes::DistributedMesh< LocalMesh >;
   using IndexType = typename MeshType::GlobalIndexType;
   using PointType = typename MeshType::PointType;
   using Array = TNL::Containers::Array< typename LocalMesh::RealType, typename LocalMesh::DeviceType, IndexType >;
   using Synchronizer = TNL::Meshes::DistributedMeshes::DistributedMeshSynchronizer< DeviceMesh, EntityType::getEntityDimension() >;

   // initialize
   DeviceMesh deviceMesh;
   deviceMesh = mesh;
   Array f( mesh.getLocalMesh().template getEntitiesCount< EntityType::getEntityDimension() >() * MeshType::getMeshDimension() );
   f.setValue( 0 );

   // set center of each local entity
   for( IndexType i = 0; i < mesh.getLocalMesh().template getEntitiesCount< EntityType >(); i++ )
      if( ! mesh.getLocalMesh().template isGhostEntity< EntityType::getEntityDimension() >( i ) ) {
         const auto center = getEntityCenter( mesh.getLocalMesh(), mesh.getLocalMesh().template getEntity< EntityType >( i ) );
         for( int d = 0; d < MeshType::getMeshDimension(); d++ )
            f.setElement( d + MeshType::getMeshDimension() * i, center[ d ] );
      }

   // synchronize
   Synchronizer sync;
   sync.initialize( deviceMesh );
   sync.synchronizeArray( f, MeshType::getMeshDimension() );

   // check all centers
   IndexType errors = 0;
   for( IndexType i = 0; i < mesh.getLocalMesh().template getEntitiesCount< EntityType >(); i++ )
      if( mesh.getLocalMesh().template isGhostEntity< EntityType::getEntityDimension() >( i ) ) {
         const PointType center = getEntityCenter( mesh.getLocalMesh(), mesh.getLocalMesh().template getEntity< EntityType >( i ) );
         PointType received;
         for( int d = 0; d < MeshType::getMeshDimension(); d++ )
            received[ d ] = f.getElement( d + MeshType::getMeshDimension() * i );
         if( received != center ) {
            IndexType cellIndexes[ 2 ] = {0, 0};
            const int numCells = getCellsForFace( mesh.getLocalMesh(), i, cellIndexes );
            std::cerr << "rank " << TNL::MPI::GetRank()
                      << ": wrong result for entity " << i << " (gid " << mesh.template getGlobalIndices< EntityType::getEntityDimension() >()[i] << ")"
                      << " of dimension = " << EntityType::getEntityDimension()
                      << ": received " << received << ", expected = " << center
                      << ", neighbor cells " << cellIndexes[0] << " " << ((numCells>1) ? cellIndexes[1] : -1)
                      << std::endl;
            errors++;
         }
      }
   if( errors > 0 )
      FAIL() << "rank " << TNL::MPI::GetRank() << ": " << errors << " errors in total." << std::endl;
}

template< typename Device, typename EntityType, typename MeshType >
void testSynchronizerOnDevice( const MeshType& mesh )
{
   testSynchronizerOnDevice_global_indices< Device, EntityType >( mesh );
   testSynchronizerOnDevice_entity_centers< Device, EntityType >( mesh );
}

template< typename Mesh >
void testSynchronizer( const Mesh& mesh )
{
   testSynchronizerOnDevice< Devices::Host, typename Mesh::Cell >( mesh );
   testSynchronizerOnDevice< Devices::Host, typename Mesh::Vertex >( mesh );
   if( mesh.template getGlobalIndices< 1 >().getSize() > 0 )
      testSynchronizerOnDevice< Devices::Host, typename Mesh::Face >( mesh );
#ifdef __CUDACC__
   testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Cell >( mesh );
   testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Vertex >( mesh );
   if( mesh.template getGlobalIndices< 1 >().getSize() > 0 )
      testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Face >( mesh );
#endif
}

TEST( DistributedMeshTest, 2D_ghostLevel0 )
{
   using GridType = TNL::Meshes::Grid< 2, float, Devices::Host, int >;
   using LocalMesh = typename GridDistributor< GridType >::LocalMeshType;
   using Mesh = DistributedMesh< LocalMesh >;
   GridType grid;
   grid.setDomain( {0, 0}, {1, 1} );
   const int nproc = TNL::MPI::GetSize();
   grid.setDimensions( nproc, nproc );
   Mesh mesh;
   GridDistributor< GridType > distributor( std::sqrt(nproc), MPI_COMM_WORLD );
   const int ghostLevels = 0;
   distributor.decompose( grid, mesh, ghostLevels );
   validateMesh( mesh, distributor, ghostLevels );
   testIteration( mesh );
}

TEST( DistributedMeshTest, 2D_ghostLevel1 )
{
   using GridType = TNL::Meshes::Grid< 2, float, Devices::Host, int >;
   using LocalMesh = typename GridDistributor< GridType >::LocalMeshType;
   using Mesh = DistributedMesh< LocalMesh >;
   GridType grid;
   grid.setDomain( {0, 0}, {1, 1} );
   const int nproc = TNL::MPI::GetSize();
   grid.setDimensions( nproc, nproc );
   Mesh mesh;
   GridDistributor< GridType > distributor( std::sqrt(nproc), MPI_COMM_WORLD );
   const int ghostLevels = 1;
   distributor.decompose( grid, mesh, ghostLevels );
   validateMesh( mesh, distributor, ghostLevels );
   testIteration( mesh );
   testSynchronizer( mesh );
}

TEST( DistributedMeshTest, 2D_ghostLevel2 )
{
   using GridType = TNL::Meshes::Grid< 2, float, Devices::Host, int >;
   using LocalMesh = typename GridDistributor< GridType >::LocalMeshType;
   using Mesh = DistributedMesh< LocalMesh >;
   GridType grid;
   grid.setDomain( {0, 0}, {1, 1} );
   const int nproc = TNL::MPI::GetSize();
   grid.setDimensions( nproc, nproc );
   Mesh mesh;
   GridDistributor< GridType > distributor( std::sqrt(nproc), MPI_COMM_WORLD );
   const int ghostLevels = 2;
   distributor.decompose( grid, mesh, ghostLevels );
   validateMesh( mesh, distributor, ghostLevels );
   testIteration( mesh );
   testSynchronizer( mesh );
}

TEST( DistributedMeshTest, PVTUWriterReader )
{
   using GridType = TNL::Meshes::Grid< 2, float, Devices::Host, int >;
   using LocalMesh = typename GridDistributor< GridType >::LocalMeshType;
   using Mesh = DistributedMesh< LocalMesh >;
   GridType grid;
   grid.setDomain( {0, 0}, {1, 1} );
   const int nproc = TNL::MPI::GetSize();
   grid.setDimensions( nproc, nproc );
   Mesh mesh;
   GridDistributor< GridType > distributor( std::sqrt(nproc), MPI_COMM_WORLD );
   const int ghostLevels = 2;
   distributor.decompose( grid, mesh, ghostLevels );

   // create a .pvtu file (only rank 0 actually writes to the file)
   const std::string baseName = "DistributedMeshTest_" + std::to_string(nproc) + "proc";
   const std::string mainFilePath = baseName + ".pvtu";
   std::string subfilePath;
   {
      std::ofstream file;
      if( TNL::MPI::GetRank() == 0 )
         file.open( mainFilePath );
      using PVTU = Meshes::Writers::PVTUWriter< LocalMesh >;
      PVTU pvtu( file );
      pvtu.template writeEntities< Mesh::getMeshDimension() >( mesh );
      if( mesh.getGhostLevels() > 0 ) {
         pvtu.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
         pvtu.template writePPointData< typename Mesh::GlobalIndexType >( "GlobalIndex" );
         pvtu.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
         pvtu.template writePCellData< typename Mesh::GlobalIndexType >( "GlobalIndex" );
      }
      subfilePath = pvtu.addPiece( mainFilePath, mesh.getCommunicator() );

      // create a .vtu file for local data
      using Writer = Meshes::Writers::VTUWriter< LocalMesh >;
      std::ofstream subfile( subfilePath );
      Writer writer( subfile );
      writer.template writeEntities< LocalMesh::getMeshDimension() >( mesh.getLocalMesh() );
      if( mesh.getGhostLevels() > 0 ) {
         writer.writePointData( mesh.vtkPointGhostTypes(), Meshes::VTK::ghostArrayName() );
         writer.writePointData( mesh.template getGlobalIndices< 0 >(), "GlobalIndex" );
         writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
         writer.writeCellData( mesh.template getGlobalIndices< 2 >(), "GlobalIndex" );
      }

      // end of scope closes the files
   }

   // load and test
   TNL::MPI::Barrier();
   Readers::PVTUReader reader( mainFilePath );
   reader.detectMesh();
   EXPECT_EQ( reader.getMeshType(), "Meshes::DistributedMesh" );
   Mesh loadedMesh;
   reader.loadMesh( loadedMesh );
   // decomposition of faces is not stored in the VTK files
   if( mesh.getGhostLevels() > 0 ) {
      distributeSubentities< 1 >( loadedMesh, /* preferHighRanks = */ false );
   }
   EXPECT_EQ( loadedMesh, mesh );

   // cleanup
   EXPECT_EQ( fs::remove( subfilePath ), true );
   TNL::MPI::Barrier();
   if( TNL::MPI::GetRank() == 0 ) {
      EXPECT_EQ( fs::remove( mainFilePath ), true );
      EXPECT_EQ( fs::remove( baseName ), true );
   }
}

} // namespace DistributedMeshTest

#endif
