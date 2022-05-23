#include <random>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
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
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
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
   {
      using CellTopology = Cell;
      using RealType = Real;
      using GlobalIndexType = GlobalIndex;
      using LocalIndexType = LocalIndex;

      static constexpr int spaceDimension = SpaceDimension;
      static constexpr int meshDimension = Cell::dimension;

      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return ( subentityDimension == 0 && entityDimension >= meshDimension - 1 )
               || subentityDimension == meshDimension - 1;
      }

      static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
      {
//         return false;
         return (entityDimension == 0 || entityDimension == meshDimension - 1) && superentityDimension >= meshDimension - 1;
      }

      static constexpr bool entityTagsStorage( int entityDimension )
      {
//         return false;
         return entityDimension == 0 || entityDimension >= meshDimension - 1;
      }

      static constexpr bool dualGraphStorage()
      {
         return true;
      }

      static constexpr int dualGraphMinCommonVertices = meshDimension;
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


// TODO
// simple mesh function without SharedPointer for the mesh
template< typename Real, typename LocalMesh, int EntitiesDimension = LocalMesh::getMeshDimension() >
struct MyMeshFunction
{
   using MeshType = LocalMesh;
   using RealType = Real;
   using DeviceType = typename LocalMesh::DeviceType;
   using IndexType = typename LocalMesh::GlobalIndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   static constexpr int getEntitiesDimension() { return EntitiesDimension; }

   static constexpr int getMeshDimension() { return LocalMesh::getMeshDimension(); }

   MyMeshFunction( const LocalMesh& localMesh )
   {
      data.setSize( localMesh.template getEntitiesCount< getEntitiesDimension() >() );
   }

   const VectorType& getData() const
   {
      return data;
   }

   VectorType& getData()
   {
      return data;
   }

   private:
      VectorType data;
};


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


// TODO: copy this to the DistributedMeshTest.h
// testing global indices is not enough - entity centers are needed to ensure that the transferred data really match the physical entities
template< typename Device, typename EntityType, typename MeshType >
void testSynchronizerOnDevice( const MeshType& mesh )
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
   if( errors > 0 ) {
      std::cerr << "rank " << TNL::MPI::GetRank() << ": " << errors << " errors in total." << std::endl;
      TNL_ASSERT_TRUE( false, "test failed" );
   }
}

template< typename Mesh >
void testSynchronizer( const Mesh& mesh )
{
   testSynchronizerOnDevice< Devices::Host, typename Mesh::Vertex >( mesh );
   testSynchronizerOnDevice< Devices::Host, typename Mesh::Cell >( mesh );
   if( mesh.template getGlobalIndices< 1 >().getSize() > 0 )
      testSynchronizerOnDevice< Devices::Host, typename Mesh::Face >( mesh );
#ifdef HAVE_CUDA
   testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Vertex >( mesh );
   testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Cell >( mesh );
   if( mesh.template getGlobalIndices< 1 >().getSize() > 0 )
      testSynchronizerOnDevice< Devices::Cuda, typename Mesh::Face >( mesh );
#endif
}


template< typename Mesh >
bool testPropagationOverFaces( const Mesh& mesh, int max_iterations )
{
   using LocalMesh = typename Mesh::MeshType;
   using Index = typename Mesh::GlobalIndexType;

   const LocalMesh& localMesh = mesh.getLocalMesh();

   using CellSynchronizer = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Mesh >;
   using FaceSynchronizer = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Mesh, Mesh::getMeshDimension() - 1 >;
   CellSynchronizer cell_sync;
   FaceSynchronizer face_sync;
   cell_sync.initialize( mesh );
   face_sync.initialize( mesh );

   using Real = int;
   MyMeshFunction< Real, LocalMesh, Mesh::getMeshDimension() >     f_K( localMesh ), f_K_test( localMesh ), f_K_test_aux( localMesh );
   MyMeshFunction< Real, LocalMesh, Mesh::getMeshDimension() - 1 > f_E( localMesh );
   f_K.getData().setValue( 0 );
   f_K_test.getData().setValue( 0 );
   f_K_test_aux.getData().setValue( 0 );
   f_E.getData().setValue( 0 );

   auto make_snapshot = [&] ( Index iteration )
   {
      // create a .pvtu file (only rank 0 actually writes to the file)
      const std::string mainFilePath = "data_" + std::to_string(iteration) + ".pvtu";
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
      pvtu.template writePCellData< Real >( "function values" );
      pvtu.template writePCellData< Real >( "test values" );
      const std::string subfilePath = pvtu.addPiece( mainFilePath, mesh.getCommunicator() );

      // create a .vtu file for local data
      using Writer = Meshes::Writers::VTUWriter< LocalMesh >;
      std::ofstream subfile( subfilePath );
      Writer writer( subfile );
      writer.writeMetadata( iteration, iteration );
      writer.template writeEntities< LocalMesh::getMeshDimension() >( localMesh );
      if( mesh.getGhostLevels() > 0 )
         writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
      writer.writeCellData( f_K.getData(), "function values" );
      writer.writeCellData( f_K_test.getData(), "test values" );
   };

   // write initial state
   make_snapshot( 0 );

   // captures for the iteration kernel
   auto f_K_view = f_K.getData().getView();
   auto f_K_test_view = f_K_test.getData().getView();
   auto f_K_test_aux_view = f_K_test_aux.getData().getView();
   auto f_E_view = f_E.getData().getView();
   Pointers::DevicePointer< const LocalMesh > localMeshDevicePointer( localMesh );
   const LocalMesh* localMeshPointer = &localMeshDevicePointer.template getData< typename LocalMesh::DeviceType >();

   const Real boundary_value = 10;

   bool all_done = false;
   int iteration = 0;
   do {
      iteration++;
      if( TNL::MPI::GetRank() == 0 )
         std::cout << "Computing iteration " << iteration << "..." << std::endl;

      const Index prev_sum = sum( f_K.getData() );

      f_K_test_aux_view = f_K_test_view;

      // iterate over all local cells
      auto testKernel = [f_K_test_aux_view, f_K_test_view, localMeshPointer] __cuda_callable__ ( Index K ) mutable
      {
         Real max = f_K_test_aux_view[ K ];
         for( int e = 0; e < localMeshPointer->template getSubentitiesCount< LocalMesh::getMeshDimension(), LocalMesh::getMeshDimension() - 1 >( K ); e++ ) {
            const Index E = localMeshPointer->template getSubentityIndex< LocalMesh::getMeshDimension(), LocalMesh::getMeshDimension() - 1 >( K, e );
            Index cellIndexes[ 2 ] = {0, 0};
            const int numCells = getCellsForFace( *localMeshPointer, E, cellIndexes );

            Real edge_value;
            if( numCells == 1 ) {
               edge_value = boundary_value;
            }
            else {
               edge_value = std::ceil( 0.5 * ( f_K_test_aux_view[ cellIndexes[ 0 ] ] + f_K_test_aux_view[ cellIndexes[ 1 ] ] ) );
//               edge_value = TNL::max( f_K_test_aux_view[ cellIndexes[ 0 ] ], f_K_test_aux_view[ cellIndexes[ 1 ] ] );
            }
            if( edge_value > max ) {
               max = edge_value;
            }
         }
         f_K_test_view[ K ] = max;
      };
      localMesh.template forLocal< LocalMesh::getMeshDimension() >( testKernel );

      // synchronize f_K_test
      cell_sync.synchronize( f_K_test );

      // iterate over all local faces
      auto faceAverageKernel = [f_K_view, f_E_view, localMeshPointer] __cuda_callable__ ( Index E ) mutable
      {
         TNL_ASSERT_FALSE( localMeshPointer->template isGhostEntity< LocalMesh::getMeshDimension() - 1 >( E ),
                           "iterator bug - got a ghost entity" );

         Index cellIndexes[ 2 ] = {0, 0};
         const int numCells = getCellsForFace( *localMeshPointer, E, cellIndexes );

         if( numCells == 1 ) {
            TNL_ASSERT_FALSE( localMeshPointer->template isGhostEntity< LocalMesh::getMeshDimension() >( cellIndexes[0] ),
                              // NOTE: c_str does not work on GPU
                              //("iterator bug - boundary face " + std::to_string(E) + " on a ghost cell "
                              // + std::to_string(cellIndexes[0])).c_str() );
                              "iterator bug - boundary face on a ghost cell" );
            f_E_view[ E ] = boundary_value;
         }
         else {
            f_E_view[ E ] = std::ceil( 0.5 * ( f_K_view[ cellIndexes[ 0 ] ] + f_K_view[ cellIndexes[ 1 ] ] ) );
//            f_E_view[ E ] = TNL::max( f_K_view[ cellIndexes[ 0 ] ], f_K_view[ cellIndexes[ 1 ] ] );
         }
      };
      localMesh.template forLocal< LocalMesh::getMeshDimension() - 1 >( faceAverageKernel );

      // synchronize f_E
      face_sync.synchronize( f_E );

      // iterate over all local cells
      auto kernel = [f_K_view, f_E_view, localMeshPointer] __cuda_callable__ ( Index K ) mutable
      {
         Real max = f_K_view[ K ];
         for( int e = 0; e < localMeshPointer->template getSubentitiesCount< LocalMesh::getMeshDimension(), LocalMesh::getMeshDimension() - 1 >( K ); e++ ) {
            const Index E = localMeshPointer->template getSubentityIndex< LocalMesh::getMeshDimension(), LocalMesh::getMeshDimension() - 1 >( K, e );
            if( f_E_view[ E ] > max ) {
               max = f_E_view[ E ];
            }
         }
         f_K_view[ K ] = max;
      };
      localMesh.template forLocal< LocalMesh::getMeshDimension() >( kernel );

      // synchronize f_K
      cell_sync.synchronize( f_K );

      // write output
      make_snapshot( iteration );

      // check correctness
      if( f_K_view != f_K_test_view ) {
         std::cerr << "ERROR: propatation over faces differs from the propagation over neighbor cells. Differing values are:\n";
         for( Index K = 0; K < f_K_view.getSize(); K++ )
            if( f_K_view[ K ] != f_K_test_view[ K ] )
               std::cerr << "   rank = " << TNL::MPI::GetRank() << ", K = " << K << ": " << f_K_view[ K ] << " instead of " << f_K_test_view[ K ] << "\n";
         std::cerr.flush();
         TNL_ASSERT_TRUE( false, "test failed" );
      }

      // check if finished
      const bool done = sum( f_K.getData() ) == prev_sum || iteration > max_iterations;
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
   config.addEntry< int >( "max-iterations", "Maximum number of iterations to compute", 100 );
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
   const int max_iterations = parameters.getParameter< int >( "max-iterations" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;

      // print basic mesh info
      mesh.printInfo( std::cout );

      // distribute faces
      TNL::Meshes::DistributedMeshes::distributeSubentities< MeshType::getMeshDimension() - 1 >( mesh );

      // test synchronizer
      testSynchronizer( mesh );

      // test simple propagation algorithm
      return testPropagationOverFaces( std::forward<MeshType>(mesh), max_iterations );
   };
   return ! Meshes::resolveAndLoadDistributedMesh< MyConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
