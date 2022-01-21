// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <string>
#include <vector>
#include <variant>

#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/VTKTraits.h>
#include <TNL/Meshes/Traits.h>

namespace TNL {
namespace Meshes {
//! \brief Namespace for mesh readers.
namespace Readers {

struct MeshReaderError
: public std::runtime_error
{
   MeshReaderError( std::string readerName, std::string msg )
   : std::runtime_error( readerName + " error: " + msg )
   {}
};

class MeshReader
{
public:
   using VariantVector = std::variant< std::vector< std::int8_t >,
                                         std::vector< std::uint8_t >,
                                         std::vector< std::int16_t >,
                                         std::vector< std::uint16_t >,
                                         std::vector< std::int32_t >,
                                         std::vector< std::uint32_t >,
                                         std::vector< std::int64_t >,
                                         std::vector< std::uint64_t >,
                                         std::vector< float >,
                                         std::vector< double > >;

   MeshReader() = default;

   MeshReader( const std::string& fileName )
   : fileName( fileName )
   {}

   virtual ~MeshReader() {}

   void setFileName( const std::string& fileName )
   {
      reset();
      this->fileName = fileName;
   }

   /**
    * \brief This method resets the reader to an empty state.
    *
    * In particular, implementations should call the \ref resetBase method to
    * reset the arrays holding the intermediate mesh representation.
    */
   virtual void reset()
   {
      resetBase();
   }

   /**
    * \brief Main method responsible for reading the mesh file.
    *
    * The implementation has to set all protected attributes of this class such
    * that the mesh representation can be loaded into the mesh object by the
    * \ref loadMesh method.
    */
   virtual void detectMesh() = 0;

   /**
    * \brief Method which loads the intermediate mesh representation into a
    * mesh object.
    *
    * This overload applies to structured grids, i.e. \ref TNL::Meshes::Grid.
    *
    * When the method exits, the intermediate mesh representation is destroyed
    * to save memory. However, depending on the specific file format, the mesh
    * file may remain open so that the user can load additional data.
    */
   template< typename MeshType >
   std::enable_if_t< isGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      // check that detectMesh has been called
      if( meshType == "" )
         detectMesh();

      // check if we have a grid
      if( meshType != "Meshes::Grid" )
         throw MeshReaderError( "MeshReader", "the file does not contain a structured grid, it is " + meshType );

      if( getMeshDimension() != mesh.getMeshDimension() )
         throw MeshReaderError( "MeshReader", "cannot load a " + std::to_string(getMeshDimension()) + "-dimensional "
                                              "grid into a mesh of type " + std::string(getType(mesh)) );

      // check that the grid attributes were set
      if( gridExtent.size() != 6 )
         throw MeshReaderError( "MeshReader", "gridExtent has invalid size: " + std::to_string(gridExtent.size()) + " (should be 6)" );
      if( gridOrigin.size() != 3 )
         throw MeshReaderError( "MeshReader", "gridOrigin has invalid size: " + std::to_string(gridOrigin.size()) + " (should be 3)" );
      if( gridSpacing.size() != 3 )
         throw MeshReaderError( "MeshReader", "gridSpacing has invalid size: " + std::to_string(gridSpacing.size()) + " (should be 3)" );

      // split the extent into begin and end
      typename MeshType::CoordinatesType begin, end;
      for( int i = 0; i < begin.getSize(); i++ ) {
         begin[i] = gridExtent[2 * i];
         end[i] = gridExtent[2 * i + 1];
      }
      mesh.setDimensions(end - begin);

      // transform the origin and calculate proportions
      typename MeshType::PointType origin, proportions;
      for( int i = 0; i < origin.getSize(); i++ ) {
         origin[i] = gridOrigin[i] + begin[i] * gridSpacing[i];
         proportions[i] = (end[i] - begin[i]) * gridSpacing[i];
      }
      mesh.setDomain( origin, proportions );
   }

   /**
    * \brief Method which loads the intermediate mesh representation into a
    * mesh object.
    *
    * This overload applies to unstructured meshes, i.e. \ref TNL::Meshes::Mesh.
    *
    * When the method exits, the intermediate mesh representation is destroyed
    * to save memory. However, depending on the specific file format, the mesh
    * file may remain open so that the user can load additional data.
    */
   template< typename MeshType >
   std::enable_if_t< ! isGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      // check that detectMesh has been called
      if( meshType == "" )
         detectMesh();

      // check if we have an unstructured mesh
      if( meshType != "Meshes::Mesh" )
         throw MeshReaderError( "MeshReader", "the file does not contain an unstructured mesh, it is " + meshType );

      // skip empty mesh (the cell shape is indeterminate)
      if( NumberOfPoints == 0 && NumberOfCells == 0 ) {
         mesh = MeshType {};
         return;
      }

      // check that the cell shape mathes
      const VTK::EntityShape meshCellShape = VTK::TopologyToEntityShape< typename MeshType::template EntityTraits< MeshType::getMeshDimension() >::EntityTopology >::shape;
      if( meshCellShape != cellShape )
         throw MeshReaderError( "MeshReader", "the mesh cell shape " + VTK::getShapeName(meshCellShape) + " does not match the shape "
                                            + "of cells used in the file (" + VTK::getShapeName(cellShape) + ")" );

      using MeshBuilder = MeshBuilder< MeshType >;
      using NeighborCountsArray = typename MeshBuilder::NeighborCountsArray;
      using PointType = typename MeshType::PointType;
      using FaceSeedType = typename MeshBuilder::FaceSeedType;
      using CellSeedType = typename MeshBuilder::CellSeedType;

      MeshBuilder meshBuilder;
      meshBuilder.setEntitiesCount( NumberOfPoints, NumberOfCells, NumberOfFaces );

      // assign points
      visit( [&meshBuilder](auto&& array) {
               PointType p;
               std::size_t i = 0;
               for( auto c : array ) {
                  int dim = i++ % 3;
                  if( dim >= PointType::getSize() )
                     continue;
                  p[dim] = c;
                  if( dim == PointType::getSize() - 1 )
                     meshBuilder.setPoint( (i - 1) / 3, p );
               }
            },
            pointsArray
         );

      // assign faces
      visit( [this, &meshBuilder](auto&& connectivity) {
               // let's just assume that the connectivity and offsets arrays have the same type...
               using std::get;
               const auto& offsets = get< std::decay_t<decltype(connectivity)> >( faceOffsetsArray );

               // Set corners counts
               NeighborCountsArray cornersCounts( NumberOfFaces );
               std::size_t offsetStart = 0;
               for( std::size_t i = 0; i < NumberOfFaces; i++ ) {
                  const std::size_t offsetEnd = offsets[ i ];
                  cornersCounts[ i ] = offsetEnd - offsetStart;
                  offsetStart = offsetEnd;
               }
               meshBuilder.setFaceCornersCounts( std::move( cornersCounts ) );

               // Set corner ids
               offsetStart = 0;
               for( std::size_t i = 0; i < NumberOfFaces; i++ ) {
                  FaceSeedType seed = meshBuilder.getFaceSeed( i );
                  const std::size_t offsetEnd = offsets[ i ];
                  for( std::size_t o = offsetStart; o < offsetEnd; o++ )
                     seed.setCornerId( o - offsetStart, connectivity[ o ] );
                  offsetStart = offsetEnd;
               }
            },
            faceConnectivityArray
         );

      // assign cells
      visit( [this, &meshBuilder](auto&& connectivity) {
               // let's just assume that the connectivity and offsets arrays have the same type...
               using std::get;
               const auto& offsets = get< std::decay_t<decltype(connectivity)> >( cellOffsetsArray );

               // Set corners counts
               NeighborCountsArray cornersCounts( NumberOfCells );
               std::size_t offsetStart = 0;
               for( std::size_t i = 0; i < NumberOfCells; i++ ) {
                  const std::size_t offsetEnd = offsets[ i ];
                  cornersCounts[ i ] = offsetEnd - offsetStart;
                  offsetStart = offsetEnd;
               }
               meshBuilder.setCellCornersCounts( std::move( cornersCounts ) );

               // Set corner ids
               offsetStart = 0;
               for( std::size_t i = 0; i < NumberOfCells; i++ ) {
                  CellSeedType seed = meshBuilder.getCellSeed( i );
                  const std::size_t offsetEnd = offsets[ i ];
                  for( std::size_t o = offsetStart; o < offsetEnd; o++ )
                     seed.setCornerId( o - offsetStart, connectivity[ o ] );
                  offsetStart = offsetEnd;
               }
            },
            cellConnectivityArray
         );

      // reset arrays since they are not needed anymore
      pointsArray = faceConnectivityArray = cellConnectivityArray = faceOffsetsArray = cellOffsetsArray = typesArray = {};

      // deduplicate faces (important for VTK formats, not for FPMA)
      if( NumberOfFaces > 0 )
         meshBuilder.deduplicateFaces();

      if( ! meshBuilder.build( mesh ) )
         throw MeshReaderError( "MeshReader", "MeshBuilder failed" );
   }

   virtual VariantVector
   readPointData( std::string arrayName )
   {
      throw Exceptions::NotImplementedError( "readPointData is not implemented in the mesh reader for this specific file format." );
   }

   virtual VariantVector
   readCellData( std::string arrayName )
   {
      throw Exceptions::NotImplementedError( "readCellData is not implemented in the mesh reader for this specific file format." );
   }

   std::string
   getMeshType() const
   {
      return meshType;
   }

   int
   getMeshDimension() const
   {
      return meshDimension;
   }

   int
   getSpaceDimension() const
   {
      return spaceDimension;
   }

   VTK::EntityShape
   getCellShape() const
   {
      return cellShape;
   }

   std::string
   getRealType() const
   {
      if( forcedRealType.empty() )
         return pointsType;
      return forcedRealType;
   }

   std::string
   getGlobalIndexType() const
   {
      if( forcedGlobalIndexType.empty() )
         return connectivityType;
      return forcedGlobalIndexType;
   }

   std::string
   getLocalIndexType() const
   {
      return forcedLocalIndexType;
   }

   void
   forceRealType( std::string realType )
   {
      forcedRealType = std::move( realType );
   }

   void
   forceGlobalIndexType( std::string globalIndexType )
   {
      forcedGlobalIndexType = std::move( globalIndexType );
   }

   void
   forceLocalIndexType( std::string localIndexType )
   {
      forcedLocalIndexType = std::move( localIndexType );
   }

protected:
   // input file name
   std::string fileName;

   // type of the mesh (either Meshes::Grid or Meshes::Mesh or Meshes::DistributedMesh)
   // (it is also an indicator that detectMesh has been successfully called)
   std::string meshType;

   // attributes of the mesh
   std::size_t NumberOfPoints, NumberOfFaces, NumberOfCells;
   int meshDimension, spaceDimension;
   VTK::EntityShape cellShape = VTK::EntityShape::Vertex;

   // string representation of mesh types (forced means specified by the user, otherwise
   // the type detected by detectMesh takes precedence)
   std::string forcedRealType = "";
   std::string forcedGlobalIndexType = "";
   std::string forcedLocalIndexType = "short int";  // not stored in any file format

   // intermediate representation of a grid (this is relevant only for TNL::Meshes::Grid)
   std::vector< std::int64_t > gridExtent;
   std::vector< double > gridOrigin, gridSpacing;

   // intermediate representation of the unstructured mesh (matches the VTU
   // file format, other formats have to be converted)
   VariantVector pointsArray, cellConnectivityArray, cellOffsetsArray,
                 faceConnectivityArray, faceOffsetsArray,
                 typesArray;

   // string representation of each array's value type
   std::string pointsType, connectivityType, offsetsType, typesType;

   void resetBase()
   {
      meshType = "";
      NumberOfPoints = NumberOfFaces = NumberOfCells = 0;
      meshDimension = spaceDimension = 0;
      cellShape = VTK::EntityShape::Vertex;

      reset_std_vectors( gridExtent, gridOrigin, gridSpacing );

      pointsArray = cellConnectivityArray = cellOffsetsArray = faceConnectivityArray = faceOffsetsArray = typesArray = {};
      pointsType = connectivityType = offsetsType = typesType = "";
   }

   template< typename T >
   void reset_std_vectors( std::vector< T >& v )
   {
      v.clear();
      v.shrink_to_fit();
   }

   template< typename T, typename... Ts >
   void reset_std_vectors( std::vector< T >& v, std::vector< Ts >&... vs )
   {
      reset_std_vectors( v );
      reset_std_vectors( vs... );
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
