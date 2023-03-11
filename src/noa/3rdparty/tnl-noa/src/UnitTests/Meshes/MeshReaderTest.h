#pragma once

#include <gtest/gtest.h>

#include <string>
#include <fstream>

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

template< typename ReaderType, template<typename> class WriterType, typename MeshType >
void test_reader( const MeshType& mesh, std::string outputFileName )
{
   // write the mesh into the file (new scope is needed to properly close the file)
   {
      // NOTE: we must open the file in binary mode to avoid LF/CRLF conversions on Windows
      std::ofstream file( outputFileName, std::ios_base::binary );
      WriterType< MeshType > writer( file );
      writer.writeEntities( mesh );
   }

   MeshType mesh_in;
   ReaderType reader( outputFileName );
   reader.loadMesh( mesh_in );

   EXPECT_EQ( mesh_in, mesh );

   EXPECT_EQ( std::remove( outputFileName.c_str() ), 0 );
}

// Test that:
// 1. resolveMeshType resolves the mesh type correctly
// 2. resolveAndLoadMesh loads the mesh
template< template<typename> class WriterType, typename ConfigTag, typename MeshType >
void test_resolveAndLoadMesh( const MeshType& mesh, std::string outputFileName, std::string globalIndexType = "auto" )
{
   // write the mesh into the file (new scope is needed to properly close the file)
   {
      // NOTE: we must open the file in binary mode to avoid LF/CRLF conversions on Windows
      std::ofstream file( outputFileName, std::ios_base::binary );
      WriterType< MeshType > writer( file );
      writer.writeEntities( mesh );
   }

   auto wrapper = [&] ( TNL::Meshes::Readers::MeshReader& reader, auto&& mesh2 )
   {
      using MeshType2 = std::decay_t< decltype(mesh2) >;

      // static_assert does not work, the wrapper is actually instantiated for all resolved types
//      static_assert( std::is_same< MeshType2, MeshType >::value, "mesh type was not resolved as expected" );
      EXPECT_EQ( std::string( TNL::getType< MeshType2 >() ), std::string( TNL::getType< MeshType >() ) );

      // operator== does not work for instantiations of the wrapper with MeshType2 != MeshType
//      EXPECT_EQ( mesh2, mesh );
      std::stringstream str1, str2;
      str1 << mesh;
      str2 << mesh2;
      EXPECT_EQ( str2.str(), str1.str() );

      return true;
   };

   const bool status = TNL::Meshes::resolveAndLoadMesh< ConfigTag, TNL::Devices::Host >( wrapper, outputFileName, "auto", "auto", globalIndexType );
   EXPECT_TRUE( status );

   EXPECT_EQ( std::remove( outputFileName.c_str() ), 0 );
}

template< typename ReaderType, template<typename> class WriterType, typename MeshType >
void test_meshfunction( const MeshType& mesh, std::string outputFileName, std::string type = "PointData" )
{
   using ArrayType = TNL::Containers::Array< std::int32_t >;
   ArrayType array_scalars, array_vectors;
   if( type == "PointData" ) {
      array_scalars.setSize( 1 * mesh.template getEntitiesCount< 0 >() );
      array_vectors.setSize( 3 * mesh.template getEntitiesCount< 0 >() );
   }
   else {
      array_scalars.setSize( 1 * mesh.template getEntitiesCount< MeshType::getMeshDimension() >() );
      array_vectors.setSize( 3 * mesh.template getEntitiesCount< MeshType::getMeshDimension() >() );
   }
   for( int i = 0; i < array_scalars.getSize(); i++ )
      array_scalars[i] = i;
   for( int i = 0; i < array_vectors.getSize(); i++ )
      array_vectors[i] = i;

   // write the mesh into the file (new scope is needed to properly close the file)
   {
      // NOTE: we must open the file in binary mode to avoid LF/CRLF conversions on Windows
      std::ofstream file( outputFileName, std::ios_base::binary );
      WriterType< MeshType > writer( file );
      writer.writeMetadata( 42, 3.14 );  // cycle, time
      writer.writeEntities( mesh );
      if( type == "PointData" ) {
         writer.writePointData( array_scalars, "foo" );
         writer.writePointData( array_vectors, "bar", 3 );
      }
      else {
         writer.writeCellData( array_scalars, "foo" );
         writer.writeCellData( array_vectors, "bar", 3 );
      }
   }

   MeshType mesh_in;
   ReaderType reader( outputFileName );
   reader.loadMesh( mesh_in );
   EXPECT_EQ( mesh_in, mesh );

   ArrayType array_scalars_in, array_vectors_in;
   typename ReaderType::VariantVector variant_scalars, variant_vectors;
   if( type == "PointData" ) {
      variant_scalars = reader.readPointData( "foo" );
      variant_vectors = reader.readPointData( "bar" );
   }
   else {
      variant_scalars = reader.readCellData( "foo" );
      variant_vectors = reader.readCellData( "bar" );
   }
   using std::visit;
   visit( [&array_scalars_in](auto&& vector) {
            array_scalars_in = vector;
         },
         variant_scalars
      );
   visit( [&array_vectors_in](auto&& vector) {
            array_vectors_in = vector;
         },
         variant_vectors
      );
   EXPECT_EQ( array_scalars_in, array_scalars );
   EXPECT_EQ( array_vectors_in, array_vectors );

   EXPECT_EQ( std::remove( outputFileName.c_str() ), 0 );
}
