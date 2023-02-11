// Implemented by: Jakub Klinkovský

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/PVTUWriter.h>
#include <TNL/Meshes/MeshDetails/IndexPermutationApplier.h>
#include <TNL/Meshes/MeshBuilder.h>

#include <metis.h>

#include <numeric>   // std::iota
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <chrono>

using namespace TNL;
using MetisIndexArray = Containers::Array< idx_t, Devices::Sequential, idx_t >;

struct DecomposeMeshConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< DecomposeMeshConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< DecomposeMeshConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< DecomposeMeshConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< DecomposeMeshConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< DecomposeMeshConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< DecomposeMeshConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< DecomposeMeshConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< DecomposeMeshConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< DecomposeMeshConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< DecomposeMeshConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< DecomposeMeshConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = GlobalIndex >
   struct MeshConfig
   : public TNL::Meshes::DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         constexpr int D = Cell::dimension;
         // subvertices of faces are needed due to cell boundary tags
         if( subentityDimension == 0 && entityDimension >= D - 1 )
            return true;
         // subfaces of cells are needed for polyhedral meshes
         if( std::is_same< Cell, TNL::Meshes::Topologies::Polyhedron >::value && subentityDimension == D - 1 && entityDimension == D )
            return true;
         return false;
      }

      static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
      {
         constexpr int D = Cell::dimension;
         // superentities from faces to cells are needed due to cell boundary tags
         return superentityDimension == D && entityDimension == D - 1;
      }

      static constexpr bool entityTagsStorage( int entityDimension )
      {
         constexpr int D = Cell::dimension;
         return entityDimension >= D - 1;
      }

      static constexpr bool dualGraphStorage()
      {
         return false;
      }
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the mesh." );
   config.addEntry< String >( "input-file-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< String >( "output-file", "Output mesh file (PVTU format)." );
   config.addRequiredEntry< unsigned >( "subdomains", "Number of subdomains to decompose the mesh." );
   config.addEntry< unsigned >( "ghost-levels", "Number of ghost levels by which the subdomains overlap.", 0 );
   config.addEntry< unsigned >( "min-common-vertices",
                                "Specifies the number of common nodes that two elements must have in order to put an "
                                "edge between them in the dual graph. By default it is equal to the mesh dimension." );
   config.addDelimiter( "METIS options:" );
   config.addEntry< String >( "metis-ptype", "Partitioning method.", "KWAY" );
      config.addEntryEnum( "KWAY" );
      config.addEntryEnum( "RB" );
   // NOTE: disabled because VOL requires the `vsize` array to be used
//   config.addEntry< String >( "metis-objtype", "Type of the objective (used only by metis-ptype=KWAY).", "CUT" );
//      config.addEntryEnum( "CUT" );
//      config.addEntryEnum( "VOL" );
   config.addEntry< String >( "metis-ctype", "Matching scheme to be used during coarsening.", "RM" );
      config.addEntryEnum( "RM" );
      config.addEntryEnum( "SHEM" );
   config.addEntry< String >( "metis-iptype", "Algorithm used during initial partitioning.", "GROW" );
      config.addEntryEnum( "GROW" );
      config.addEntryEnum( "RANDOM" );
      config.addEntryEnum( "EDGE" );
      config.addEntryEnum( "NODE" );
   config.addEntry< String >( "metis-rtype", "Algorithm used for refinement.", "FM" );
      config.addEntryEnum( "FM" );
      config.addEntryEnum( "GREEDY" );
      config.addEntryEnum( "SEP2SIDED" );
      config.addEntryEnum( "SEP1SIDED" );
   config.addEntry< int >( "metis-no2hop",
                           "Specifies that the coarsening will not perform any 2–hop matchings when the standard "
                           "matching approach fails to sufficiently coarsen the graph. The 2–hop matching is very "
                           "effective for graphs with power-law degree distributions. "
                           "0 - Performs a 2–hop matching. 1 - Does not perform a 2–hop matching. ",
                           0 );
      config.addEntryEnum( 0 );
      config.addEntryEnum( 1 );
   config.addEntry< unsigned >( "metis-ncuts",
                                "Specifies the number of different partitionings that it will compute. The final "
                                "partitioning is the one that achieves the best edgecut or communication volume.",
                                1 );
   config.addEntry< unsigned >( "metis-niter",
                                "Specifies the number of iterations for the refinement algorithms at each stage of the "
                                "uncoarsening process.",
                                10 );
   config.addEntry< unsigned >( "metis-ufactor",
                                "Specifies the maximum allowed load imbalance among the partitions. "
                                "The default value is 1 for metis-ptype=RB and 30 for metis-ptype=KWAY." );
   config.addEntry< int >( "metis-minconn",
                           "Specifies that the partitioning routines should try to minimize the maximum degree of the "
                           "subdomain graph. Note that the option applies only to metis-ptype=KWAY.",
                           1 );
      config.addEntryEnum( 0 );
      config.addEntryEnum( 1 );
   config.addEntry< int >( "metis-contig",
                           "Specifies that the partitioning routines should try to produce partitions that are "
                           "contiguous. Note that if the input graph is not connected this option is ignored. "
                           "Note that the option applies only to metis-ptype=KWAY.",
                           1 );
      config.addEntryEnum( 0 );
      config.addEntryEnum( 1 );
   config.addEntry< unsigned >( "metis-dbglvl",
                                "Specifies the amount of progress/debugging information will be printed during the execution "
                                "of the algorithms. The default value is 0 (no debugging/progress information). A non-zero "
                                "value can be supplied that is obtained by a bit-wise OR of the following values.\n"
                                "   METIS_DBG_INFO (1)         Prints various diagnostic messages.\n"
                                "   METIS_DBG_TIME (2)         Performs timing analysis.\n"
                                "   METIS_DBG_COARSEN (4)      Displays various statistics during coarsening.\n"
                                "   METIS_DBG_REFINE (8)       Displays various statistics during refinement.\n"
                                "   METIS_DBG_IPART (16)       Displays various statistics during initial partitioning.\n"
                                "   METIS_DBG_MOVEINFO (32)    Displays detailed information about vertex moves during refinement.\n"
                                "   METIS_DBG_SEPINFO (64)     Displays information about vertex separators.\n"
                                "   METIS_DBG_CONNINFO (128)   Displays information related to the minimization of subdomain connectivity.\n"
                                "   METIS_DBG_CONTIGINFO (256) Displays information related to the elimination of connected components.",
                                0 );
}

void setMETISoptions( idx_t options[METIS_NOPTIONS], const Config::ParameterContainer& parameters )
{
   // partitioning method
   const String ptype = parameters.getParameter< String >( "metis-ptype" );
   if( ptype == "KWAY" )
      options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
   if( ptype == "RB" )
      options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
   // type of the objective (used only by METIS_PTYPE_KWAY)
   options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // or METIS_OBJTYPE_VOL (requires vsize to be used)
   // matching scheme to be used during coarsening
   const String ctype = parameters.getParameter< String >( "metis-ctype" );
   if( ctype == "RM" )
      options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
   if( ctype == "SHEM" )
      options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
   // algorithm used during initial partitioning
   const String iptype = parameters.getParameter< String >( "metis-iptype" );
   if( iptype == "GROW" )
      options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
   if( iptype == "RANDOM" )
      options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_RANDOM;
   if( iptype == "EDGE" )
      options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;
   if( iptype == "NODE" )
      options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;
   // algorithm used for refinement
   const String rtype = parameters.getParameter< String >( "metis-rtype" );
   if( rtype == "FM" )
      options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
   if( rtype == "GREEDY" )
      options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
   if( rtype == "SEP2SIDED" )
      options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP2SIDED;
   if( rtype == "SEP1SIDED" )
      options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;
   // Specifies that the coarsening will not perform any 2–hop matchings when the standard
   // matching approach fails to sufficiently coarsen the graph. The 2–hop matching is very
   // effective for graphs with power-law degree distributions.
   // 0 - Performs a 2–hop matching. 1 - Does not perform a 2–hop matching.
   options[METIS_OPTION_NO2HOP] = parameters.getParameter< int >( "metis-no2hop" );
   // Specifies the number of different partitionings that it will compute. The final
   // partitioning is the one that achieves the best edgecut or communication volume.
   // Default is 1.
   options[METIS_OPTION_NCUTS] = parameters.getParameter< unsigned >( "metis-ncuts" );
   // Specifies the number of iterations for the refinement algorithms at each stage of the
   // uncoarsening process. Default is 10.
   options[METIS_OPTION_NITER] = parameters.getParameter< unsigned >( "metis-niter" );
   // Specifies the maximum allowed load imbalance among the partitions.
   // The default value is 1 for ptype=rb and 30 for ptype=kway.
   if( parameters.checkParameter( "metis-ufactor" ) )
      options[METIS_OPTION_UFACTOR] = parameters.getParameter< unsigned >( "metis-ufactor" );
   // Specifies that the partitioning routines should try to minimize the maximum degree of the
   // subdomain graph, i.e., the graph in which each partition is a node, and edges connect
   // subdomains with a shared interface.
   // 0 - Does not explicitly minimize the maximum connectivity.
   // 1 - Explicitly minimize the maximum connectivity.
   // NOTE: applies only to METIS_PTYPE_KWAY
   options[METIS_OPTION_MINCONN] = parameters.getParameter< int >( "metis-minconn" );
   // Specifies that the partitioning routines should try to produce partitions that are
   // contiguous. Note that if the input graph is not connected this option is ignored.
   // NOTE: applies only to METIS_PTYPE_KWAY
   options[METIS_OPTION_CONTIG] = parameters.getParameter< int >( "metis-contig" );
   // seed for the random number generator
   options[METIS_OPTION_SEED] = std::chrono::system_clock::now().time_since_epoch().count();
   // numbering scheme for the adjacency structure of a graph or element-node structure of a mesh
   // (0 is C-style, 1 is Fortran-style)
   options[METIS_OPTION_NUMBERING] = 0;
   // Specifies the amount of progress/debugging information will be printed during the execution
   // of the algorithms. The default value is 0 (no debugging/progress information). A non-zero
   // value can be supplied that is obtained by a bit-wise OR of the following values.
   //    METIS_DBG_INFO (1)         Prints various diagnostic messages.
   //    METIS_DBG_TIME (2)         Performs timing analysis.
   //    METIS_DBG_COARSEN (4)      Displays various statistics during coarsening.
   //    METIS_DBG_REFINE (8)       Displays various statistics during refinement.
   //    METIS_DBG_IPART (16)       Displays various statistics during initial partitioning.
   //    METIS_DBG_MOVEINFO (32)    Displays detailed information about vertex moves during refinement.
   //    METIS_DBG_SEPINFO (64)     Displays information about vertex separators.
   //    METIS_DBG_CONNINFO (128)   Displays information related to the minimization of subdomain connectivity.
   //    METIS_DBG_CONTIGINFO (256) Displays information related to the elimination of connected components.
   options[METIS_OPTION_DBGLVL] = parameters.getParameter< unsigned >( "metis-dbglvl" );
}

template< typename Mesh >
void
decompose_and_save( const Mesh& mesh,
                    const unsigned nparts,
                    const MetisIndexArray& part,
                    const std::shared_ptr< idx_t >& dual_xadj,
                    const std::shared_ptr< idx_t >& dual_adjncy,
                    const unsigned ncommon,
                    const unsigned ghost_levels,
                    const std::string pvtuFileName )
{
   using Index = typename Mesh::GlobalIndexType;
   using IndexArray = Containers::Array< Index, Devices::Sequential, Index >;

   constexpr bool is_polyhedral = std::is_same_v< typename Mesh::Config::CellTopology, Meshes::Topologies::Polyhedron >;

   const Index cellsCount = mesh.template getEntitiesCount< typename Mesh::Cell >();
   const Index pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();

   // count cells in each subdomain
   IndexArray cells_counts( nparts );
   cells_counts.setValue( 0 );
   for( Index i = 0; i < cellsCount; i++ )
      ++cells_counts[ part[ i ] ];

   // build offsets for the partitioned cell indices
   IndexArray cells_offsets( nparts );
   cells_offsets[0] = 0;
   for( unsigned p = 1; p < nparts; p++ )
      cells_offsets[p] = cells_offsets[p-1] + cells_counts[p-1];

   // construct block-wise local-to-global mapping for cells
   IndexArray cells_local_to_global( cellsCount );
   {
      IndexArray offsets; offsets = cells_offsets;
      for( Index i = 0; i < cellsCount; i++ ) {
         const Index p = part[ i ];
         cells_local_to_global[ offsets[p]++ ] = i;
      }
   }

   auto is_ghost_neighbor = [&] ( const typename Mesh::Cell& cell )
   {
      const Index neighbors_start = dual_xadj.get()[ cell.getIndex() ];
      const Index neighbors_end = dual_xadj.get()[ cell.getIndex() + 1 ];
      for( Index i = neighbors_start; i < neighbors_end; i++ ) {
         const Index neighbor_idx = dual_adjncy.get()[ i ];
         if( part[ cell.getIndex() ] != part[ neighbor_idx ] )
            return true;
      }
      return false;
   };

   // construct global index permutation for cells
   // convention: seed index = global index in the partitioned mesh,
   // cell index = global index in the original mesh
   IndexArray seed_to_cell_index( cellsCount );
   for( unsigned p = 0; p < nparts; p++ ) {
      Index assigned = 0;
      std::set< Index > boundary;
      std::set< Index > ghost_neighbors;
      for( Index local_idx = 0; local_idx < cells_counts[ p ]; local_idx++ ) {
         const Index global_idx = cells_local_to_global[ cells_offsets[ p ] + local_idx ];
         const auto& cell = mesh.template getEntity< typename Mesh::Cell >( global_idx );
         // check global domain boundary first
         if( mesh.template isBoundaryEntity< Mesh::getMeshDimension() >( cell.getIndex() ) )
            boundary.insert( global_idx );
         // check subdomain boundary
         else if( is_ghost_neighbor( cell ) )
            ghost_neighbors.insert( global_idx );
         // otherwise subdomain interior - assign index now
         else
            seed_to_cell_index[ cells_offsets[ p ] + assigned++ ] = global_idx;
      }
      for( auto global_idx : boundary )
         seed_to_cell_index[ cells_offsets[ p ] + assigned++ ] = global_idx;
      for( auto global_idx : ghost_neighbors )
         seed_to_cell_index[ cells_offsets[ p ] + assigned++ ] = global_idx;
      TNL_ASSERT_EQ( assigned, cells_counts[ p ], "bug in the global index permutation generator" );
   }
   cells_local_to_global.reset();
   // cell_to_seed_index is an inverse permutation of seed_to_cell_index
   IndexArray cell_to_seed_index( cellsCount );
   for( Index i = 0; i < cellsCount; i++ )
      cell_to_seed_index[ seed_to_cell_index[ i ] ] = i;

   // construct global index permutation for points
   // convention: points at subdomain boundaries are assigned to the subdomain with the higher number
   IndexArray point_old_to_new_global_index( pointsCount );
   IndexArray points_counts( nparts );
   points_counts.setValue( 0 );
   {
      // first assign points to subdomains - the subdomain with the highest number takes the point
      // (go over local cells, set subvertex owner = cell owner, higher rank will overwrite)
      IndexArray point_to_subdomain( pointsCount );
      for( unsigned p = 0; p < nparts; p++ ) {
         for( Index local_idx = 0; local_idx < cells_counts[ p ]; local_idx++ ) {
            const Index global_idx = seed_to_cell_index[ cells_offsets[ p ] + local_idx ];
            const auto& cell = mesh.template getEntity< typename Mesh::Cell >( global_idx );
            const Index subvertices = cell.template getSubentitiesCount< 0 >();
            for( Index j = 0; j < subvertices; j++ ) {
               const Index v = cell.template getSubentityIndex< 0 >( j );
               point_to_subdomain[ v ] = p;
            }
         }
      }
      // assign global indices to points
      Index pointIdx = 0;
      for( unsigned p = 0; p < nparts; p++ ) {
         for( Index local_idx = 0; local_idx < cells_counts[ p ]; local_idx++ ) {
            const Index global_idx = seed_to_cell_index[ cells_offsets[ p ] + local_idx ];
            const auto& cell = mesh.template getEntity< typename Mesh::Cell >( global_idx );
            const Index subvertices = cell.template getSubentitiesCount< 0 >();
            for( Index j = 0; j < subvertices; j++ ) {
               const Index v = cell.template getSubentityIndex< 0 >( j );
               if( point_to_subdomain[ v ] == (Index) p ) {
                  // assign index
                  point_old_to_new_global_index[ v ] = pointIdx++;
                  // mark as assigned
                  point_to_subdomain[ v ] = nparts;
                  // increase count
                  ++points_counts[ p ];
               }
            }
         }
      }
   }

   // build offsets for the partitioned point indices
   IndexArray points_offsets( nparts );
   points_offsets[0] = 0;
   for( unsigned p = 1; p < nparts; p++ )
      points_offsets[p] = points_offsets[p-1] + points_counts[p-1];

   // write a .pvtu file
   using PVTU = Meshes::Writers::PVTUWriter< Mesh >;
   std::ofstream file( pvtuFileName );
   PVTU pvtu( file );
   pvtu.template writeEntities< Mesh::getMeshDimension() >( Mesh{}, ghost_levels, ncommon );
   if( ghost_levels > 0 ) {
      // the PointData and CellData from the individual files should be added here
      pvtu.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
      pvtu.template writePPointData< Index >( "GlobalIndex" );
      pvtu.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
      pvtu.template writePCellData< Index >( "GlobalIndex" );
   }

   std::cout << "Writing subdomains..." << std::endl;
   for( unsigned p = 0; p < nparts; p++ ) {
      const std::string outputFileName = pvtu.addPiece( pvtuFileName, p );
      std::cout << outputFileName << std::endl;

      // Due to ghost levels, we don't know the number of cells, let alone points, in each
      // subdomain ahead of time. Hence, we use dynamic data structures instead of MeshBuilder.

      // We'll need global-to-local index mapping for vertices (and faces in case of polyhedral meshes).
      // Here we also record which points (and faces) are actually needed.
      std::map< Index, Index > vertex_global_to_local;
      std::map< Index, Index > face_global_to_local;

      // Cell seeds refer usually to vertices, except for polyhedral meshes where they refer to faces
      using CellSeed = typename Mesh::MeshTraitsType::CellSeedType;
      std::vector< CellSeed > cell_seeds;
      std::vector< Index > cell_seeds_global_indices;
      std::set< Index > cell_global_indices;

      // Face seeds are needed only for polyhedral meshes
      using FaceSeed = typename Mesh::MeshTraitsType::FaceSeedType;
      std::vector< FaceSeed > face_seeds;

      auto add_face = [&] ( const auto& face )
      {
         if( face_global_to_local.count( face.getIndex() ) != 0 )
            return;
         FaceSeed seed;
         seed.setCornersCount( face.template getSubentitiesCount< 0 >() );
         for( Index v = 0; v < face.template getSubentitiesCount< 0 >(); v++ ) {
            const Index global_idx = face.template getSubentityIndex< 0 >( v );
            if( vertex_global_to_local.count(global_idx) == 0 )
               vertex_global_to_local.insert( {global_idx, vertex_global_to_local.size()} );
            seed.setCornerId( v, vertex_global_to_local[ global_idx ] );
         }
         face_seeds.push_back( seed );
         face_global_to_local.insert( {face.getIndex(), face_global_to_local.size()} );
      };

      auto add_cell = [&] ( const auto& cell )
      {
         if( cell_global_indices.count( cell.getIndex() ) != 0 )
            return;
         CellSeed seed;
         if constexpr( is_polyhedral ) {
            seed.setCornersCount( cell.template getSubentitiesCount< Mesh::getMeshDimension() - 1 >() );
            for( Index f = 0; f < cell.template getSubentitiesCount< Mesh::getMeshDimension() - 1 >(); f++ ) {
               const Index global_idx = cell.template getSubentityIndex< Mesh::getMeshDimension() - 1 >( f );
               const auto& face = mesh.template getEntity< Mesh::getMeshDimension() - 1 >( global_idx );
               add_face( face );
               seed.setCornerId( f, face_global_to_local[ face.getIndex() ] );
            }
         }
         else {
            seed.setCornersCount( cell.template getSubentitiesCount< 0 >() );
            for( Index v = 0; v < cell.template getSubentitiesCount< 0 >(); v++ ) {
               const Index global_idx = cell.template getSubentityIndex< 0 >( v );
               if( vertex_global_to_local.count(global_idx) == 0 )
                  vertex_global_to_local.insert( {global_idx, vertex_global_to_local.size()} );
               seed.setCornerId( v, vertex_global_to_local[ global_idx ] );
            }
         }
         cell_seeds.push_back( seed );
         cell_seeds_global_indices.push_back( cell_to_seed_index[ cell.getIndex() ] );
         cell_global_indices.insert( cell.getIndex() );
      };

      // iterate over local cells, add only local points (to ensure that ghost points are ordered after all local points)
      for( Index local_idx = 0; local_idx < cells_counts[ p ]; local_idx++ ) {
         const Index global_idx = seed_to_cell_index[ cells_offsets[ p ] + local_idx ];
         const auto& cell = mesh.template getEntity< typename Mesh::Cell >( global_idx );
         for( Index v = 0; v < cell.template getSubentitiesCount< 0 >(); v++ ) {
            const Index global_vert_idx = cell.template getSubentityIndex< 0 >( v );
            if( point_old_to_new_global_index[global_vert_idx] >= points_offsets[p] &&
                point_old_to_new_global_index[global_vert_idx] < points_offsets[p] + points_counts[p] )
            {
               if( vertex_global_to_local.count(global_vert_idx) == 0 )
                  vertex_global_to_local.insert( {global_vert_idx, vertex_global_to_local.size()} );
            }
         }
      }
      TNL_ASSERT_EQ( (Index) vertex_global_to_local.size(), points_counts[p],
                     "some local points were not added in the first pass" );
      // TODO: in case of a polyhedral mesh, now we should first add local faces (to ensure that ghost faces are ordered after all local faces)
      // iterate over local cells, create seeds and record ghost neighbor indices
      std::vector< Index > ghost_neighbors;
      for( Index local_idx = 0; local_idx < cells_counts[ p ]; local_idx++ ) {
         const Index global_idx = seed_to_cell_index[ cells_offsets[ p ] + local_idx ];
         const auto& cell = mesh.template getEntity< typename Mesh::Cell >( global_idx );
         add_cell( cell );
         if( is_ghost_neighbor( cell ) )
            ghost_neighbors.push_back( global_idx );
      }

      // collect seed indices of ghost cells
      std::set< Index > ghost_seed_indices;
      for( unsigned gl = 0; gl < ghost_levels; gl++ ) {
         std::vector< Index > new_ghosts;
         for( auto global_idx : ghost_neighbors ) {
            const Index neighbors_start = dual_xadj.get()[ global_idx ];
            const Index neighbors_end = dual_xadj.get()[ global_idx + 1 ];
            for( Index i = neighbors_start; i < neighbors_end; i++ ) {
               const Index neighbor_idx = dual_adjncy.get()[ i ];
               // skip neighbors on the local subdomain
               if( part[ neighbor_idx ] == (int) p )
                  continue;
               const Index neighbor_seed_idx = cell_to_seed_index[ neighbor_idx ];
               // skip neighbors whose seed was already added
               if( ghost_seed_indices.count( neighbor_seed_idx ) == 0 ) {
                  new_ghosts.push_back( neighbor_idx );
                  ghost_seed_indices.insert( neighbor_seed_idx );
               }
            }
         }
         std::swap( ghost_neighbors, new_ghosts );
      }
      ghost_neighbors.clear();
      ghost_neighbors.shrink_to_fit();

      // add ghost cells (the set is sorted by the seed index)
      for( auto idx : ghost_seed_indices ) {
         // the ghost_seed_indices array may contain duplicates and even local
         // cells, but add_cell takes care of uniqueness, so we don't have to
         // care about that
         const auto& cell = mesh.template getEntity< typename Mesh::Cell >( seed_to_cell_index[ idx ] );
         add_cell( cell );
      }
      ghost_seed_indices.clear();
      cell_global_indices.clear();
      face_global_to_local.clear();

      // set points needed for the subdomain
      using PointArrayType = typename Mesh::MeshTraitsType::PointArrayType;
      PointArrayType points( vertex_global_to_local.size() );
      // create "GlobalIndex" PointData array
      IndexArray pointsGlobalIndices( vertex_global_to_local.size() );
      for( auto it : vertex_global_to_local ) {
         points[ it.second ] = mesh.getPoint( it.first );
         pointsGlobalIndices[ it.second ] = point_old_to_new_global_index[ it.first ];
      }
      vertex_global_to_local.clear();

      // create "GlobalIndex" CellData array
      IndexArray cellsGlobalIndices = cell_seeds_global_indices;
      cell_seeds_global_indices.clear();
      cell_seeds_global_indices.shrink_to_fit();

      // create "vtkGhostType" CellData and PointData arrays - see https://blog.kitware.com/ghost-and-blanking-visibility-changes/
      Containers::Array< std::uint8_t, Devices::Sequential, Index > cellGhosts( cell_seeds.size() );
      Containers::Array< std::uint8_t, Devices::Sequential, Index > pointGhosts( points.getSize() );
      for( Index i = 0; i < cells_counts[ p ]; i++ )
         cellGhosts[ i ] = 0;
      for( Index i = cells_counts[ p ]; i < (Index) cell_seeds.size(); i++ )
         cellGhosts[ i ] = (std::uint8_t) Meshes::VTK::CellGhostTypes::DUPLICATECELL;
      // point ghosts are more tricky because they were assigned to the subdomain with higher number
      Index pointsGhostCount = 0;
      for( Index i = 0; i < points.getSize(); i++ ) {
         const Index global_idx = pointsGlobalIndices[ i ];
         if( global_idx < points_offsets[ p ] || global_idx >= points_offsets[ p ] + points_counts[ p ] ) {
            pointGhosts[ i ] = (std::uint8_t) Meshes::VTK::PointGhostTypes::DUPLICATEPOINT;
            pointsGhostCount++;
         }
         else
            pointGhosts[ i ] = 0;
      }

      // reorder ghost points to make sure that global indices are sorted
      {
         // prepare vector with an identity permutation
         std::vector< Index > permutation( points.getSize() );
         std::iota( permutation.begin(), permutation.end(), (Index) 0 );

         // sort the subarray corresponding to ghost entities by the global index
         std::stable_sort( permutation.begin() + points.getSize() - pointsGhostCount,
                           permutation.end(),
                           [&pointsGlobalIndices](auto& left, auto& right) {
            return pointsGlobalIndices[ left ] < pointsGlobalIndices[ right ];
         });

         // copy the permutation into TNL array
         typename Mesh::GlobalIndexArray perm( permutation );
         permutation.clear();
         permutation.shrink_to_fit();

         // apply the permutation
         using PermutationApplier = TNL::Meshes::IndexPermutationApplier< Mesh, 0 >;
         // - pointGhosts
         PermutationApplier::permuteArray( pointGhosts, perm );
         // - pointsGlobalIndices
         PermutationApplier::permuteArray( pointsGlobalIndices, perm );
         // - points
         PermutationApplier::permuteArray( points, perm );
         // - seeds.setCornerID (inverse perm)
         std::vector< Index > iperm( points.getSize() );
         for( Index i = 0; i < perm.getSize(); i++ )
            iperm[ perm[ i ] ] = i;
         if constexpr( is_polyhedral ) {
            for( auto& seed : face_seeds ) {
               auto& cornerIds = seed.getCornerIds();
               for( Index v = 0; v < cornerIds.getSize(); v++ )
                  cornerIds[ v ] = iperm[ cornerIds[ v ] ];
            }
         }
         else {
            for( auto& seed : cell_seeds ) {
               auto& cornerIds = seed.getCornerIds();
               for( Index v = 0; v < cornerIds.getSize(); v++ )
                  cornerIds[ v ] = iperm[ cornerIds[ v ] ];
            }
         }
      }

      // initialize the MeshBuilder
      TNL::Meshes::MeshBuilder< Mesh > builder;
      builder.setEntitiesCount( points.getSize(), cell_seeds.size(), face_seeds.size() );

      // copy points to the MeshBuilder
      for( Index i = 0; i < points.getSize(); i++ )
         builder.setPoint( i, points[ i ] );
      points.reset();

      // copy face seeds to the MeshBuilder
      if constexpr( is_polyhedral ) {
         typename TNL::Meshes::MeshBuilder< Mesh >::NeighborCountsArray corners_counts( face_seeds.size() );
         for( std::size_t i = 0; i < face_seeds.size(); i++ )
            corners_counts[ i ] = face_seeds[ i ].getCornersCount();
         builder.setFaceCornersCounts( corners_counts );
         for( std::size_t i = 0; i < face_seeds.size(); i++ ) {
            const auto& cornerIds = face_seeds[ i ].getCornerIds();
            for( Index v = 0; v < cornerIds.getSize(); v++ )
               builder.getFaceSeed( i ).setCornerId( v, cornerIds[ v ] );
         }
         face_seeds.clear();
         face_seeds.shrink_to_fit();
      }

      // copy cell seeds to the MeshBuilder
      if constexpr( TNL::Meshes::Topologies::IsDynamicTopology< typename Mesh::Cell::EntityTopology >::value ) {
         typename TNL::Meshes::MeshBuilder< Mesh >::NeighborCountsArray corners_counts( cell_seeds.size() );
         for( std::size_t i = 0; i < cell_seeds.size(); i++ )
            corners_counts[ i ] = cell_seeds[ i ].getCornersCount();
         builder.setCellCornersCounts( corners_counts );
      }
      for( std::size_t i = 0; i < cell_seeds.size(); i++ ) {
         const auto& cornerIds = cell_seeds[ i ].getCornerIds();
         for( Index v = 0; v < cornerIds.getSize(); v++ )
            builder.getCellSeed( i ).setCornerId( v, cornerIds[ v ] );
      }
      cell_seeds.clear();
      cell_seeds.shrink_to_fit();

      // init mesh for the subdomain
      Mesh subdomain;
      if( ! builder.build( subdomain ) )
         throw std::runtime_error( "mesh builder failed for subdomain " + std::to_string(p) );

      // write the subdomain
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( subdomain );
      if( ghost_levels > 0 ) {
         writer.writePointData( pointGhosts, Meshes::VTK::ghostArrayName() );
         writer.writePointData( pointsGlobalIndices, "GlobalIndex" );
         writer.writeCellData( cellGhosts, Meshes::VTK::ghostArrayName() );
         writer.writeCellData( cellsGlobalIndices, "GlobalIndex" );
      }
   }
}

template< typename Mesh >
void run( const Mesh& mesh, const Config::ParameterContainer& parameters )
{
   using Index = typename Mesh::GlobalIndexType;

   // warn if input mesh has 64-bit indices, but METIS uses only 32-bit indices
   if( IDXTYPEWIDTH == 32 && sizeof(Index) > 4 )
      std::cerr << "Warning: the input mesh uses 64-bit indices, but METIS was compiled only with 32-bit indices. "
                   "Decomposition may not work correctly if the index values overflow the 32-bit type." << std::endl;

   // get the mesh connectivity information in a format suitable for METIS. Actually, the same
   // format is used by the XML-based VTK formats - the only difference is that METIS requires
   // `offsets` to start with 0.
   std::vector< idx_t > connectivity;
   std::vector< idx_t > offsets;
   offsets.push_back(0);
   const Index cellsCount = mesh.template getEntitiesCount< typename Mesh::Cell >();
   for( Index i = 0; i < cellsCount; i++ ) {
      const auto& entity = mesh.template getEntity< typename Mesh::Cell >( i );
      const Index subvertices = entity.template getSubentitiesCount< 0 >();
      for( Index j = 0; j < subvertices; j++ )
         connectivity.push_back( entity.template getSubentityIndex< 0 >( j ) );
      offsets.push_back( connectivity.size() );
   }

   // number of elements (cells)
   idx_t ne = mesh.template getEntitiesCount< typename Mesh::Cell >();
   // number of nodes (vertices)
   idx_t nn = mesh.template getEntitiesCount< 0 >();
   // pointers to arrays storing the mesh in a CSR-like format
   idx_t* eptr = offsets.data();
   idx_t* eind = connectivity.data();
   // Specifies the number of common nodes that two elements must have in order to put an edge
   // between them in the dual graph.
   idx_t ncommon = Mesh::getMeshDimension();
   if( parameters.checkParameter( "min-common-vertices" ) )
      ncommon = parameters.template getParameter< unsigned >( "min-common-vertices" );
   // numbering scheme for the adjacency structure of a graph or element-node structure of a mesh
   // (0 is C-style, 1 is Fortran-style)
   idx_t numflag = 0;
   // These arrays store the adjacency structure of the generated dual graph. The format is of
   // the adjacency structure is described in Section 5.5 of the METIS manual. Memory for these
   // arrays is allocated by METIS’ API using the standard malloc function. It is the
   // responsibility of the application to free this memory by calling free. METIS provides the
   // METIS_Free that is a wrapper to C’s free function.
   idx_t* xadj = nullptr;
   idx_t* adjncy = nullptr;

   // We could use METIS_PartMeshDual directly instead of METIS_MeshToDual + METIS_PartGraph*,
   // but we need to reuse the dual graph for the generation of ghost cells.
   std::cout << "Running METIS_MeshToDual..." << std::endl;
   int status = METIS_MeshToDual(&ne, &nn, eptr, eind, &ncommon, &numflag, &xadj, &adjncy);

   // wrap xadj and adjncy with shared_ptr
   std::shared_ptr<idx_t> shared_xadj {xadj, METIS_Free};
   std::shared_ptr<idx_t> shared_adjncy {adjncy, METIS_Free};

   switch( status )
   {
      case METIS_OK: break;
      case METIS_ERROR_INPUT:
         throw std::runtime_error( "METIS_MeshToDual failed due to an input error." );
      case METIS_ERROR_MEMORY:
         throw std::runtime_error( "METIS_MeshToDual failed due to a memory allocation error." );
      case METIS_ERROR:
      default:
         throw std::runtime_error( "METIS_MeshToDual failed with an unspecified error." );
   }

   // The number of vertices in the graph.
   idx_t nvtxs = ne;
   // The number of balancing constraints. It should be at least 1.
   idx_t ncon = 1;
   // An array of size `ne` specifying the weights of the elements. A NULL value can be passed
   // to indicate that all elements have an equal weight.
   idx_t* vwgt = nullptr;
   // An array of size `ne` specifying the size of the elements that is used for computing the
   // total communication volume as described in Section 5.7 of the METIS manual. A NULL value
   // can be passed when the objective is cut or when all elements have an equal size.
   idx_t* vsize = nullptr;
   // The weights of the edges as described in Section 5.5 of the METIS manual.
   idx_t* adjwgt = nullptr;  // METIS_PartMeshDual uses NULL too
   // The number of parts to partition the mesh.
   idx_t nparts = parameters.template getParameter< unsigned >( "subdomains" );
   // This is an array of size nparts that specifies the desired weight for each partition. The
   // target partition weight for the i-th partition is specified at tpwgts[i] (the numbering for
   // the partitions starts from 0). The sum of the tpwgts[] entries must be 1.0. A NULL value
   // can be passed to indicate that the graph should be equally divided among the partitions.
   real_t* tpwgts = nullptr;
   // This is an array of size ncon that specifies the allowed load imbalance tolerance for each
   // constraint. For the i-th partition and j-th constraint the allowed weight is the
   // ubvec[j]*tpwgts[i*ncon+j] fraction of the j-th’s constraint total weight. The load
   // imbalances must be greater than 1.0. A NULL value can be passed indicating that the load
   // imbalance tolerance for each constraint should be 1.001 (for ncon=1) or 1.01 (for ncon>1).
   real_t* ubvec = nullptr;  // METIS_PartMeshDual uses NULL too
   // Upon successful completion, this variable stores either the edgecut or the total
   // communication volume of the dual graph’s partitioning.
   idx_t objval = 0;
   // Array of size `ne` that upon successful completion stores the partition array for the
   // elements of the mesh.
   MetisIndexArray part_array( nvtxs );
   idx_t* part = part_array.getData();

   // Array of METIS options as described in Section 5.4 of the METIS manual.
   idx_t options[METIS_NOPTIONS];
   // future-proof (or just in case we forgot to set some options explicitly)
   METIS_SetDefaultOptions(options);
   // set METIS options from parameters
   setMETISoptions(options, parameters);

   if( nparts == 1 ) {
      // k-way partitioning from Metis fails for nparts == 1 (segfault),
      // RB succeeds but produces nonsense
      part_array.setValue( 0 );
   }
   else {
      if( options[METIS_OPTION_PTYPE] == METIS_PTYPE_KWAY ) {
         std::cout << "Running METIS_PartGraphKway..." << std::endl;
         status = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt, &nparts, tpwgts, ubvec, options, &objval, part);
      }
      else {
         std::cout << "Running METIS_PartGraphRecursive..." << std::endl;
         status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt, &nparts, tpwgts, ubvec, options, &objval, part);
      }

      switch( status )
      {
         case METIS_OK: break;
         case METIS_ERROR_INPUT:
            throw std::runtime_error( "METIS_PartGraph failed due to an input error." );
         case METIS_ERROR_MEMORY:
            throw std::runtime_error( "METIS_PartGraph failed due to a memory allocation error." );
         case METIS_ERROR:
         default:
            throw std::runtime_error( "METIS_PartGraph failed with an unspecified error." );
      }
   }

   // deallocate auxiliary vectors
   connectivity.clear();
   connectivity.shrink_to_fit();
   offsets.clear();
   offsets.shrink_to_fit();

   const unsigned ghost_levels = parameters.getParameter< unsigned >( "ghost-levels" );
   const std::string pvtuFileName = parameters.template getParameter< String >( "output-file" );
   decompose_and_save( mesh, nparts, part_array, shared_xadj, shared_adjncy, ncommon, ghost_levels, pvtuFileName );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String inputFileFormat = parameters.getParameter< String >( "input-file-format" );
   const String outputFile = parameters.template getParameter< String >( "output-file" );
   if( ! outputFile.endsWith( ".pvtu" ) ) {
      std::cerr << "Error: the output file must have a '.pvtu' extension." << std::endl;
      return EXIT_FAILURE;
   }

   auto wrapper = [&] ( const auto& reader, auto&& mesh )
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      run( std::forward<MeshType>(mesh), parameters );
      return true;
   };
   const bool status = Meshes::resolveAndLoadMesh< DecomposeMeshConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
