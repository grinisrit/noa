// Implemented by: Ján Bobot, Jakub Klinkovský

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>
#include <TNL/Meshes/Geometry/getDecomposedMesh.h>
#include <TNL/Meshes/Geometry/getPlanarMesh.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Readers/MeshReader.h>
#include <TNL/Meshes/Topologies/IsDynamicTopology.h>
#include <TNL/Benchmarks/Benchmarks.h>

#include "MeshConfigs.h"
#include "MemoryInfo.h"

using namespace TNL;
using namespace TNL::Meshes;
using namespace TNL::Meshes::Readers;
using namespace TNL::Benchmarks;

template< typename Device >
bool checkDevice( const Config::ParameterContainer& parameters )
{
   const String device = parameters.getParameter< String >( "devices" );
   if( device == "all" )
      return true;
   if( std::is_same< Device, Devices::Host >::value && device == "host" )
      return true;
   if( std::is_same< Device, Devices::Cuda >::value && device == "cuda" )
      return true;
   return false;
}

std::string removeNamespaces( const String & topology )
{
  std::size_t found = topology.find_last_of("::");
  return topology.substr( found + 1 );
}

template< typename Mesh >
struct MeshBenchmarks
{
   static_assert( std::is_same< typename Mesh::DeviceType, Devices::Host >::value, "The mesh should be loaded on the host." );

   static bool run( Benchmark<> & benchmark, const Config::ParameterContainer & parameters )
   {
      Logging::MetadataColumns metadataColumns = {
         // {"mesh-file", meshFile},
         {"config", Mesh::Config::getConfigType()},
         //{"topology", removeNamespaces( getType< typename Mesh::Config::CellTopology >() ) },
         //{"space dim", std::to_string( Mesh::Config::spaceDimension )},
         //{"real", getType< typename Mesh::RealType >()},
         //{"gid_t", getType< typename Mesh::GlobalIndexType >()},
         //{"lid_t", getType< typename Mesh::LocalIndexType >()}
      };
      benchmark.setMetadataColumns( metadataColumns );

      const String & meshFile = parameters.getParameter< String >( "mesh-file" );
      auto reader = getMeshReader( meshFile, "auto" );
      Mesh mesh;

      try {
         reader->loadMesh( mesh );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load mesh from file '" << meshFile << "'." << std::endl;
         return false;
      }

      dispatchTests( benchmark, parameters, mesh, reader );

      return true;
   }

   static void dispatchTests( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const Mesh & mesh, std::shared_ptr< MeshReader > reader )
   {
      ReaderDispatch::exec( benchmark, parameters, reader );
      InitDispatch::exec( benchmark, parameters, reader );
      DecompositionDispatch::exec( benchmark, parameters, mesh );
      PlanarDispatch::exec( benchmark, parameters, mesh );
      MeasuresDispatch::exec( benchmark, parameters, mesh );
      MemoryDispatch::exec( benchmark, parameters, mesh );
      CopyDispatch::exec( benchmark, parameters, mesh );
   }

   struct ReaderDispatch
   {
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, std::shared_ptr< MeshReader > reader )
      {
         benchmark.setOperation( String( "Reader" ) );
         benchmark_reader( benchmark, parameters, reader );
      }
   };

   struct InitDispatch
   {
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, std::shared_ptr< MeshReader > reader )
      {
         benchmark.setOperation( String( "Init" ) );
         benchmark_init( benchmark, parameters, reader );
      }
   };

   struct DecompositionDispatch
   {
      // Polygonal Mesh
      template< typename M,
                std::enable_if_t< std::is_same< typename M::Config::CellTopology, Topologies::Polygon >::value, bool > = true >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
         benchmark.setOperation( String( "Decomposition (c)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToCentroid >( benchmark, parameters, mesh_src );

         benchmark.setOperation( String( "Decomposition (p)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToPoint >( benchmark, parameters, mesh_src );
      }

      // Polyhedral Mesh
      template< typename M,
                std::enable_if_t< std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value, bool  > = true >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
         benchmark.setOperation( String( "Decomposition (cc)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                  EntityDecomposerVersion::ConnectEdgesToCentroid >( benchmark, parameters, mesh_src );

         benchmark.setOperation( String( "Decomposition (cp)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                  EntityDecomposerVersion::ConnectEdgesToPoint >( benchmark, parameters, mesh_src );

         benchmark.setOperation( String( "Decomposition (pc)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToPoint,
                                  EntityDecomposerVersion::ConnectEdgesToCentroid >( benchmark, parameters, mesh_src );

         benchmark.setOperation( String( "Decomposition (pp)" ) );
         benchmark_decomposition< EntityDecomposerVersion::ConnectEdgesToPoint,
                                  EntityDecomposerVersion::ConnectEdgesToPoint >( benchmark, parameters, mesh_src );
      }

      // Other than Polygonal and Polyhedral Mesh
      template< typename M,
                std::enable_if_t< ! std::is_same< typename M::Config::CellTopology, Topologies::Polygon >::value &&
                                  ! std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value, bool  > = true >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
      }
   };

   struct PlanarDispatch
   {
      template< typename M,
                std::enable_if_t< M::Config::spaceDimension == 3 &&
                                 (std::is_same< typename M::Config::CellTopology, Topologies::Polygon >::value ||
                                  std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value ), bool > = true >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
         benchmark.setOperation( String( "Planar Correction (c)" ) );
         benchmark_planar< EntityDecomposerVersion::ConnectEdgesToCentroid >( benchmark, parameters, mesh_src );

         benchmark.setOperation( String( "Planar Correction (p)" ) );
         benchmark_planar< EntityDecomposerVersion::ConnectEdgesToPoint >( benchmark, parameters, mesh_src );
      }

      template< typename M,
                std::enable_if_t< M::Config::spaceDimension < 3 ||
                                 (! std::is_same< typename M::Config::CellTopology, Topologies::Polygon >::value &&
                                  ! std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value ), bool > = true >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
      }
   };

   struct MeasuresDispatch
   {
      template< typename M >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh )
      {
         benchmark.setOperation( String("Measures") );
         benchmark_measures< Devices::Host >( benchmark, parameters, mesh );
         #ifdef HAVE_CUDA
         benchmark_measures< Devices::Cuda >( benchmark, parameters, mesh );
         #endif
      }
   };

   struct MemoryDispatch
   {
      template< typename M >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer& parameters, const M& mesh_src )
      {
         benchmark.setOperation( String("Memory") );
         benchmark_memory( benchmark, parameters, mesh_src );
      }
   };

   struct CopyDispatch
   {
      template< typename M >
      static void exec( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
      {
         #ifdef HAVE_CUDA
         benchmark.setOperation( String("Copy CPU->GPU") );
         benchmark_copy< Devices::Host, Devices::Cuda >( benchmark, parameters, mesh_src );
         benchmark.setOperation( String("Copy GPU->CPU") );
         benchmark_copy< Devices::Cuda, Devices::Host >( benchmark, parameters, mesh_src );
         #endif
      }
   };

   static void benchmark_reader( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, std::shared_ptr< MeshReader > reader )
   {
      if( ! checkDevice< Devices::Host >( parameters ) )
         return;

      auto reset = [&]() {
         reader->reset();
      };

      auto benchmark_func = [&] () {
         reader->detectMesh();
      };

      benchmark.time< Devices::Host >( reset,
                                       "CPU",
                                       benchmark_func );
   }

   static void benchmark_init( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, std::shared_ptr< MeshReader > reader )
   {
      if( ! checkDevice< Devices::Host >( parameters ) )
         return;

      auto reset = [&]() {
         reader->detectMesh();
      };

      auto benchmark_func = [&] () {
         Mesh mesh;
         reader->loadMesh( mesh );
      };

      benchmark.time< Devices::Host >( reset,
                                       "CPU",
                                       benchmark_func );
   }

   template< EntityDecomposerVersion DecomposerVersion,
             EntityDecomposerVersion SubDecomposerVersion = EntityDecomposerVersion::ConnectEdgesToPoint,
             typename M >
   static void benchmark_decomposition( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
   {
      // skip benchmarks on devices which the user did not select
      if( ! checkDevice< Devices::Host >( parameters ) )
         return;

      auto benchmark_func = [&] () {
         auto meshBuilder = decomposeMesh< DecomposerVersion, SubDecomposerVersion >( mesh_src );
      };

      benchmark.time< Devices::Host >( "CPU",
                                       benchmark_func );
   }

   template< EntityDecomposerVersion DecomposerVersion,
             typename M,
             std::enable_if_t< M::Config::spaceDimension == 3 &&
                              (std::is_same< typename M::Config::CellTopology, Topologies::Polygon >::value ||
                               std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value ), bool > = true >
   static void benchmark_planar( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
   {
      if( ! checkDevice< Devices::Host >( parameters ) )
         return;

      auto benchmark_func = [&] () {
         auto meshBuilder = planarCorrection< DecomposerVersion >( mesh_src );
      };

      benchmark.time< Devices::Host >( "CPU",
                                       benchmark_func );
   }

   template< typename Device,
             typename M >
   static void benchmark_measures( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
   {
      using Real = typename M::RealType;
      using Index = typename M::GlobalIndexType;
      using DeviceMesh = Meshes::Mesh< typename M::Config, Device >;

      // skip benchmarks on devices which the user did not select
      if( ! checkDevice< Device >( parameters ) )
         return;

      const Index entitiesCount = mesh_src.template getEntitiesCount< M::getMeshDimension() >();

      const DeviceMesh mesh = mesh_src;
      Pointers::DevicePointer< const DeviceMesh > meshPointer( mesh );
      Containers::Array< Real, Device, Index > measures;
      measures.setSize( entitiesCount );

      auto kernel_measures = [] __cuda_callable__
         ( Index i,
           const DeviceMesh* mesh,
           Real* array )
      {
         const auto& entity = mesh->template getEntity< M::getMeshDimension() >( i );
         array[ i ] = getEntityMeasure( *mesh, entity );
      };

      auto reset = [&]() {
         measures.setValue( 0.0 );
      };

      auto benchmark_func = [&] () {
         Algorithms::ParallelFor< Device >::exec(
               (Index) 0, entitiesCount,
               kernel_measures,
               &meshPointer.template getData< Device >(),
               measures.getData() );
      };

      benchmark.time< Device >( reset,
                                (std::is_same< Device, Devices::Host >::value) ? "CPU" : "GPU",
                                benchmark_func );
   }

   template< typename M >
   static void benchmark_memory( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
   {
      if( ! checkDevice< Devices::Host >( parameters ) )
            return;

      MemoryBenchmarkResult memResult = testMemoryUsage( parameters, mesh_src );
      auto noop = [](){};
      benchmark.time< TNL::Devices::Host >( "CPU", noop, memResult );
   }

   template< typename DeviceFrom,
             typename DeviceTo,
             typename M >
   static void benchmark_copy( Benchmark<> & benchmark, const Config::ParameterContainer & parameters, const M & mesh_src )
   {
      using MeshFrom = Meshes::Mesh< typename M::Config, DeviceFrom >;
      using MeshTo = Meshes::Mesh< typename M::Config, DeviceTo >;
      using Device = typename std::conditional_t< std::is_same< DeviceFrom, Devices::Host >::value &&
                                                  std::is_same< DeviceTo, Devices::Host >::value,
                                                  Devices::Host,
                                                  Devices::Cuda >;

      // skip benchmarks on devices which the user did not select
      if( ! checkDevice< Device >( parameters ) )
         return;

      const MeshFrom meshFrom = mesh_src;

      auto benchmark_func = [&] () {
         MeshTo meshTo = meshFrom;
      };

      benchmark.time< Device >( [] () {},
                                (std::is_same< Device, Devices::Host >::value) ? "CPU" : "GPU",
                                benchmark_func );
   }
};

template< template< typename, int, typename, typename, typename > class ConfigTemplate,
          typename CellTopology,
          int SpaceDimension,
          typename Real,
          typename GlobalIndex,
          typename LocalIndex >
struct MeshBenchmarksRunner
{
    static bool
    run( Benchmark<> & benchmark,
         const Config::ParameterContainer & parameters )
   {
      using Config = ConfigTemplate< CellTopology, SpaceDimension, Real, GlobalIndex, LocalIndex >;
      using MeshType = Mesh< Config, Devices::Host >;
      return MeshBenchmarks< MeshType >::run( benchmark, parameters );
   }
};
