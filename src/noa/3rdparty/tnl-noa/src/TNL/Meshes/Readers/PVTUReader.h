// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <filesystem>
#include <utility>

#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Utils.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/VTUReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/layers/EntityTags/Traits.h>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

class PVTUReader : public XMLVTK
{
   std::string
   getSourcePath( const std::string& source )
   {
      namespace fs = std::filesystem;
      return fs::path( fileName ).parent_path() / source;
   }

#ifdef HAVE_TINYXML2
   void
   readParallelUnstructuredGrid()
   {
      using namespace tinyxml2;

      // read GhostLevel attribute
      ghostLevels = getAttributeInteger( datasetElement, "GhostLevel" );
      // read MinCommonVertices attribute (TNL-specific, optional)
      minCommonVertices = getAttributeInteger( datasetElement, "MinCommonVertices", 0 );

      // read points info
      const XMLElement* points = getChildSafe( datasetElement, "PPoints" );
      const XMLElement* pointsData = verifyHasOnlyOneChild( points, "PDataArray" );
      verifyDataArray( pointsData, "PDataArray" );
      const std::string pointsDataName = getAttributeString( pointsData, "Name" );
      if( pointsDataName != "Points" )
         throw MeshReaderError( "PVTUReader",
                                "the <PPoints> tag does not contain a <PDataArray> with Name=\"Points\" attribute" );
      pointsType = VTKDataTypes.at( getAttributeString( pointsData, "type" ) );

      // read pieces info
      const XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      while( piece != nullptr ) {
         const std::string source = getAttributeString( piece, "Source" );
         if( ! source.empty() ) {
            pieceSources.push_back( getSourcePath( source ) );
         }
         else
            throw MeshReaderError( "PVTUReader", "the Source attribute of a <Piece> element was found empty." );
         // find next
         piece = piece->NextSiblingElement( "Piece" );
      }
      if( pieceSources.empty() )
         throw MeshReaderError( "PVTUReader", "the file does not contain any <Piece> element." );

      // check that the number of pieces matches the number of MPI ranks
      const int nproc = communicator.size();
      if( (int) pieceSources.size() != nproc )
         throw MeshReaderError( "PVTUReader",
                                "the number of subdomains does not match the number of MPI ranks ("
                                   + std::to_string( pieceSources.size() ) + " vs " + std::to_string( nproc ) + ")." );

      // read the local piece source
      const int rank = communicator.rank();
      localReader.setFileName( pieceSources[ rank ] );
      localReader.detectMesh();

      // copy attributes from the local reader
      spaceDimension = localReader.getSpaceDimension();
      meshDimension = localReader.getMeshDimension();
      cellShape = localReader.getCellShape();
      pointsType = localReader.getRealType();
      connectivityType = offsetsType = localReader.getGlobalIndexType();
      typesType = "std::uint8_t";

      // TODO: assert that all MPI ranks have the same attributes

      if( ghostLevels > 0 ) {
         // load the vtkGhostType arrays from PointData and CellData
         pointTags = localReader.readPointData( VTK::ghostArrayName() );
         cellTags = localReader.readCellData( VTK::ghostArrayName() );

         // load the GlobalIndex arrays from PointData and CellData
         pointGlobalIndices = localReader.readPointData( "GlobalIndex" );
         cellGlobalIndices = localReader.readCellData( "GlobalIndex" );
      }
   }
#endif

public:
   PVTUReader() = default;

   PVTUReader( const std::string& fileName, MPI::Comm communicator = MPI_COMM_WORLD )
   : XMLVTK( fileName ), communicator( std::move( communicator ) )
   {}

   void
   setCommunicator( const MPI::Comm& communicator )
   {
      this->communicator = communicator;
   }

   void
   detectMesh() override
   {
#ifdef HAVE_TINYXML2
      reset();
      try {
         openVTKFile();
      }
      catch( const MeshReaderError& ) {
         reset();
         throw;
      }

      // verify file type
      if( fileType == "PUnstructuredGrid" )
         readParallelUnstructuredGrid();
      else
         throw MeshReaderError(
            "PVTUReader", "the reader cannot read data of the type " + fileType + ". Use a different reader if possible." );

      // indicate success by setting the mesh type
      meshType = "Meshes::DistributedMesh";
#else
      throw_no_tinyxml();
#endif
   }

   template< typename MeshType >
   std::enable_if_t< isDistributedGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      throw MeshReaderError( "MeshReader", "the PVTU reader cannot be used to load a distributed structured grid." );
   }

   template< typename MeshType >
   std::enable_if_t< ! isDistributedGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      // check that detectMesh has been called
      if( meshType.empty() )
         detectMesh();

      // check if we have a distributed unstructured mesh
      if( meshType != "Meshes::DistributedMesh" )
         throw MeshReaderError( "MeshReader", "the file does not contain an unstructured mesh, it is " + meshType );

      // load the local mesh
      auto& localMesh = mesh.getLocalMesh();
      localReader.loadMesh( localMesh );

      using Index = typename MeshType::GlobalIndexType;
      const Index pointsCount = mesh.getLocalMesh().template getEntitiesCount< 0 >();
      const Index cellsCount = mesh.getLocalMesh().template getEntitiesCount< MeshType::getMeshDimension() >();

      // set ghost levels
      mesh.setGhostLevels( ghostLevels );
      // check MinCommonVertices
      if( minCommonVertices > 0 && minCommonVertices != MeshType::Config::dualGraphMinCommonVertices )
         std::cerr << "WARNING: the mesh was decomposed with different MinCommonVertices value than the value set in the mesh "
                      "configuration "
                      "("
                   << minCommonVertices << " vs " << MeshType::Config::dualGraphMinCommonVertices << ")." << std::endl;

      if( ghostLevels > 0 ) {
         // assign point ghost tags
         using std::get;
         const std::vector< std::uint8_t > pointTags = get< std::vector< std::uint8_t > >( this->pointTags );
         if( (Index) pointTags.size() != pointsCount )
            throw MeshReaderError(
               "PVTUReader", "the vtkGhostType array in PointData has wrong size: " + std::to_string( pointTags.size() ) );
         mesh.vtkPointGhostTypes() = pointTags;
         for( Index i = 0; i < pointsCount; i++ )
            if( pointTags[ i ] & (std::uint8_t) VTK::PointGhostTypes::DUPLICATEPOINT )
               localMesh.template addEntityTag< 0 >( i, EntityTags::GhostEntity );

         // assign cell ghost tags
         using std::get;
         const std::vector< std::uint8_t > cellTags = get< std::vector< std::uint8_t > >( this->cellTags );
         if( (Index) cellTags.size() != cellsCount )
            throw MeshReaderError( "PVTUReader",
                                   "the vtkGhostType array in CellData has wrong size: " + std::to_string( cellTags.size() ) );
         mesh.vtkCellGhostTypes() = cellTags;
         for( Index i = 0; i < cellsCount; i++ ) {
            if( cellTags[ i ] & (std::uint8_t) VTK::CellGhostTypes::DUPLICATECELL )
               localMesh.template addEntityTag< MeshType::getMeshDimension() >( i, EntityTags::GhostEntity );
         }

         // update the entity tags layers after setting ghost indices
         mesh.getLocalMesh().template updateEntityTagsLayer< 0 >();
         mesh.getLocalMesh().template updateEntityTagsLayer< MeshType::getMeshDimension() >();

         // assign global indices
         auto& points_indices = mesh.template getGlobalIndices< 0 >();
         auto& cells_indices = mesh.template getGlobalIndices< MeshType::getMeshDimension() >();
         auto assign_variant_vector = []( auto& array, const VariantVector& variant_vector, Index expectedSize )
         {
            using std::visit;
            visit(
               [ &array, expectedSize ]( auto&& vector )
               {
                  if( (Index) vector.size() != expectedSize )
                     throw MeshReaderError( "PVTUReader",
                                            "the GlobalIndex array has wrong size: " + std::to_string( vector.size() )
                                               + " (expected " + std::to_string( expectedSize ) + ")." );
                  array.setSize( vector.size() );
                  std::size_t idx = 0;
                  for( auto v : vector )
                     array[ idx++ ] = v;
               },
               variant_vector );
         };
         assign_variant_vector( points_indices, pointGlobalIndices, pointsCount );
         assign_variant_vector( cells_indices, cellGlobalIndices, cellsCount );
      }

      // reset arrays since they are not needed anymore
      this->pointTags = this->cellTags = pointGlobalIndices = cellGlobalIndices = {};

      // check if we need to split the communicator
      const Index minCount = MPI::reduce( TNL::min( pointsCount, cellsCount ), MPI_MIN );
      if( minCount == 0 ) {
         // split the communicator, remove the ranks which did not get a subdomain
         const int color = ( pointsCount > 0 && cellsCount > 0 ) ? 0 : MPI_UNDEFINED;
         MPI::Comm subCommunicator = communicator.split( color, 0 );

         // set the communicator
         mesh.setCommunicator( std::move( subCommunicator ) );
      }
      else {
         // set the communicator
         mesh.setCommunicator( communicator );
      }
   }

   VariantVector
   readPointData( std::string arrayName ) override
   {
      return localReader.readPointData( arrayName );
   }

   VariantVector
   readCellData( std::string arrayName ) override
   {
      return localReader.readCellData( arrayName );
   }

   void
   reset() override
   {
      resetBase();
      ghostLevels = 0;
      reset_std_vectors( pieceSources );
      localReader.reset();
      pointTags = cellTags = pointGlobalIndices = cellGlobalIndices = {};
   }

protected:
   MPI::Comm communicator;

   int ghostLevels = 0;
   int minCommonVertices = 0;
   std::vector< std::string > pieceSources;

   VTUReader localReader;

   // additinal arrays we need to read from the localReader
   VariantVector pointTags, cellTags, pointGlobalIndices, cellGlobalIndices;
};

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
