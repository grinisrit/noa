// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunctionView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/VectorFieldGnuplotWriter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTKWriter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTUWriter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTIWriter.h>

namespace noa::TNL {
namespace Functions {

template< int Size, typename Function >
class VectorField : public Functions::Domain< Function::getDomainDimension(), Function::getDomainType() >
{
public:
   typedef Function FunctionType;
   typedef typename FunctionType::RealType RealType;
   typedef typename FunctionType::PointType PointType;

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         FunctionType::configSetup( config, prefix + convertToString( i ) + "-" );
   }

   template< typename MeshPointer >
   bool
   setup( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         if( ! vectorField[ i ].setup( parameters, prefix + convertToString( i ) + "-" ) ) {
            std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
            return false;
         }
      return true;
   }

   __cuda_callable__
   const FunctionType&
   operator[]( int i ) const
   {
      return this->vectorField[ i ];
   }

   __cuda_callable__
   FunctionType&
   operator[]( int i )
   {
      return this->vectorField[ i ];
   }

protected:
   Containers::StaticArray< Size, FunctionType > vectorField;
};

template< int Size, typename Mesh, int MeshEntityDimension, typename Real >
class VectorField< Size, MeshFunction< Mesh, MeshEntityDimension, Real > >
: public Functions::Domain< MeshFunction< Mesh, MeshEntityDimension, Real >::getDomainDimension(),
                            MeshFunction< Mesh, MeshEntityDimension, Real >::getDomainType() >
{
public:
   typedef Mesh MeshType;
   typedef Real RealType;
   typedef Pointers::SharedPointer< MeshType > MeshPointer;
   typedef MeshFunction< MeshType, MeshEntityDimension, RealType > FunctionType;
   typedef Pointers::SharedPointer< FunctionType > FunctionPointer;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::GlobalIndexType IndexType;
   typedef Containers::StaticVector< Size, RealType > VectorType;

   static constexpr int
   getEntitiesDimension()
   {
      return FunctionType::getEntitiesDimension();
   }

   static constexpr int
   getMeshDimension()
   {
      return MeshType::getMeshDimension();
   }

   static constexpr int
   getVectorDimension()
   {
      return Size;
   }

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         FunctionType::configSetup( config, prefix + convertToString( i ) + "-" );
   }

   VectorField(){};

   VectorField( const MeshPointer& meshPointer )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorField[ i ]->setMesh( meshPointer );
   };

   void
   setMesh( const MeshPointer& meshPointer )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorField[ i ]->setMesh( meshPointer );
   }

   template< typename Device = Devices::Host >
   __cuda_callable__
   const MeshType&
   getMesh() const
   {
      return this->vectorField[ 0 ].template getData< Device >().template getMesh< Device >();
   }

   const MeshPointer&
   getMeshPointer() const
   {
      return this->vectorField[ 0 ]->getMeshPointer();
   }

   bool
   setup( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         if( ! vectorField[ i ].setup( meshPointer, parameters, prefix + convertToString( i ) + "-" ) ) {
            std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
            return false;
         }
      return true;
   }

   static IndexType
   getDofs( const MeshPointer& meshPointer )
   {
      return Size * FunctionType::getDofs( meshPointer );
   }

   __cuda_callable__
   const FunctionPointer&
   operator[]( int i ) const
   {
      return this->vectorField[ i ];
   }

   __cuda_callable__
   FunctionPointer&
   operator[]( int i )
   {
      return this->vectorField[ i ];
   }

   __cuda_callable__
   void
   setElement( const IndexType i, const VectorType& v )
   {
      for( int j = 0; j < Size; j++ )
         ( *this )[ j ]->getData().setElement( i, v[ j ] );
   }

   VectorType
   getElement( const IndexType i ) const
   {
      VectorType v;
      for( int j = 0; j < Size; j++ )
         v[ j ] = ( *this )[ j ]->getData().getElement( i );
      return v;
   }

   __cuda_callable__
   VectorType
   getVector( const IndexType index ) const
   {
      VectorType v;
      for( int i = 0; i < Size; i++ )
         // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
         v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()[ index ];
      return v;
   }

   template< typename EntityType >
   void
   setValue( const EntityType& meshEntity, const FunctionType& value )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorfield[ i ].setValue( meshEntity.getIndex(), value[ i ] );
   }

   template< typename EntityType >
   __cuda_callable__
   VectorType
   getVector( const EntityType& meshEntity ) const
   {
      VectorType v;
      for( int i = 0; i < Size; i++ )
         // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
         v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()( meshEntity );
      return v;
   }

   bool
   write( const std::string& functionName, const std::string& fileName, const std::string& fileFormat = "auto" ) const
   {
      std::fstream file;
      file.open( fileName, std::ios::out );
      if( ! file ) {
         std::cerr << "Unable to open a file " << fileName << "." << std::endl;
         return false;
      }

      namespace fs = std::filesystem;
      std::string format = fileFormat;
      if( format == "auto" ) {
         format = fs::path( fileName ).extension();
         if( format.length() > 0 )
            // remove dot from the extension
            format = format.substr( 1 );
      }

      if( format == "vtk" || format == "vtu" || format == "vti" ) {
         // copy all values from the vector field into a contiguous array
         using BufferType = Containers::Array< typename VectorField::RealType, Devices::Host, IndexType >;
         const IndexType entitiesCount = getMeshPointer()->template getEntitiesCount< getEntitiesDimension() >();
         BufferType buffer( 3 * entitiesCount );
         IndexType k = 0;
         for( IndexType i = 0; i < entitiesCount; i++ ) {
            const VectorType vector = getElement( i );
            static_assert( getVectorDimension() <= 3, "VTK formats support only up to 3D vector fields." );
            for( int j = 0; j < 3; j++ )
               buffer[ k++ ] = ( j < vector.getSize() ? vector[ j ] : 0 );
         }

         if( format == "vtk" ) {
            Meshes::Writers::VTKWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
         else if( format == "vtu" ) {
            Meshes::Writers::VTUWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
         else if( format == "vti" ) {
            Meshes::Writers::VTIWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
      }
      else if( format == "gnuplot" || format == "gplt" || format == "plt" )
         return VectorFieldGnuplotWriter< VectorField >::write( *this, file );
      else {
         std::cerr << "Unknown output format: " << format << std::endl;
         return false;
      }
      return true;
   }

protected:
   Containers::StaticArray< Size, FunctionPointer > vectorField;
};

template< int Size, typename Mesh, int MeshEntityDimension, typename Real >
class VectorField< Size, MeshFunctionView< Mesh, MeshEntityDimension, Real > >
: public Functions::Domain< MeshFunctionView< Mesh, MeshEntityDimension, Real >::getDomainDimension(),
                            MeshFunctionView< Mesh, MeshEntityDimension, Real >::getDomainType() >
{
public:
   typedef Mesh MeshType;
   typedef Real RealType;
   typedef Pointers::SharedPointer< MeshType > MeshPointer;
   typedef MeshFunctionView< MeshType, MeshEntityDimension, RealType > FunctionType;
   typedef Pointers::SharedPointer< FunctionType > FunctionPointer;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename MeshType::GlobalIndexType IndexType;
   typedef Containers::StaticVector< Size, RealType > VectorType;

   static constexpr int
   getEntitiesDimension()
   {
      return FunctionType::getEntitiesDimension();
   }

   static constexpr int
   getMeshDimension()
   {
      return MeshType::getMeshDimension();
   }

   static constexpr int
   getVectorDimension()
   {
      return Size;
   }

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         FunctionType::configSetup( config, prefix + convertToString( i ) + "-" );
   }

   VectorField(){};

   VectorField( const MeshPointer& meshPointer )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorField[ i ]->setMesh( meshPointer );
   };

   void
   setMesh( const MeshPointer& meshPointer )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorField[ i ]->setMesh( meshPointer );
   }

   template< typename Device = Devices::Host >
   __cuda_callable__
   const MeshType&
   getMesh() const
   {
      return this->vectorField[ 0 ].template getData< Device >().template getMesh< Device >();
   }

   const MeshPointer&
   getMeshPointer() const
   {
      return this->vectorField[ 0 ]->getMeshPointer();
   }

   bool
   setup( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      for( int i = 0; i < Size; i++ )
         if( ! vectorField[ i ].setup( meshPointer, parameters, prefix + convertToString( i ) + "-" ) ) {
            std::cerr << "Unable to setup " << i << "-th coordinate of the vector field." << std::endl;
            return false;
         }
      return true;
   }

   static IndexType
   getDofs( const MeshPointer& meshPointer )
   {
      return Size * FunctionType::getDofs( meshPointer );
   }

   void
   bind( VectorField& vectorField )
   {
      for( int i = 0; i < Size; i++ ) {
         this->vectorField[ i ]->bind( vectorField[ i ] );
      }
   };

   template< typename Vector >
   void
   bind( const MeshPointer& meshPointer, Vector& data, IndexType offset = 0 )
   {
      TNL_ASSERT_GE( data.getSize(),
                     offset + Size * this->vectorField[ 0 ]->getDofs( meshPointer ),
                     "Attempt to bind vector which is not large enough." );
      for( int i = 0; i < Size; i++ ) {
         this->vectorField[ i ]->bind( meshPointer, data, offset );
         offset += this->vectorField[ i ]->getDofs( meshPointer );
      }
   }

   template< typename Vector >
   void
   bind( const MeshPointer& meshPointer, Pointers::SharedPointer< Vector >& dataPtr, IndexType offset = 0 )
   {
      TNL_ASSERT_GE( dataPtr->getSize(),
                     offset + Size * this->vectorField[ 0 ]->getDofs( meshPointer ),
                     "Attempt to bind vector which is not large enough." );
      for( int i = 0; i < Size; i++ ) {
         this->vectorField[ i ]->bind( meshPointer, dataPtr, offset );
         offset += this->vectorField[ i ]->getDofs( meshPointer );
      }
   }

   __cuda_callable__
   const FunctionPointer&
   operator[]( int i ) const
   {
      return this->vectorField[ i ];
   }

   __cuda_callable__
   FunctionPointer&
   operator[]( int i )
   {
      return this->vectorField[ i ];
   }

   __cuda_callable__
   void
   setElement( const IndexType i, const VectorType& v )
   {
      for( int j = 0; j < Size; j++ )
         ( *this )[ j ]->getData().setElement( i, v[ j ] );
   }

   VectorType
   getElement( const IndexType i ) const
   {
      VectorType v;
      for( int j = 0; j < Size; j++ )
         v[ j ] = ( *this )[ j ]->getData().getElement( i );
      return v;
   }

   __cuda_callable__
   VectorType
   getVector( const IndexType index ) const
   {
      VectorType v;
      for( int i = 0; i < Size; i++ )
         // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
         v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()[ index ];
      return v;
   }

   template< typename EntityType >
   void
   setValue( const EntityType& meshEntity, const FunctionType& value )
   {
      for( int i = 0; i < Size; i++ )
         this->vectorfield[ i ].setValue( meshEntity.getIndex(), value[ i ] );
   }

   template< typename EntityType >
   __cuda_callable__
   VectorType
   getVector( const EntityType& meshEntity ) const
   {
      VectorType v;
      for( int i = 0; i < Size; i++ )
         // FIXME: fix the dereferencing operator in smart pointers to be __cuda_callable__
         v[ i ] = this->vectorField[ i ].template getData< Devices::Cuda >()( meshEntity );
      return v;
   }

   bool
   write( const std::string& functionName, const std::string& fileName, const std::string& fileFormat = "auto" ) const
   {
      std::fstream file;
      file.open( fileName, std::ios::out );
      if( ! file ) {
         std::cerr << "Unable to open a file " << fileName << "." << std::endl;
         return false;
      }

      namespace fs = std::filesystem;
      std::string format = fileFormat;
      if( format == "auto" ) {
         format = fs::path( fileName ).extension();
         if( format.length() > 0 )
            // remove dot from the extension
            format = format.substr( 1 );
      }

      if( format == "vtk" || format == "vtu" || format == "vti" ) {
         // copy all values from the vector field into a contiguous array
         using BufferType = Containers::Array< typename VectorField::RealType, Devices::Host, IndexType >;
         const IndexType entitiesCount = getMeshPointer()->template getEntitiesCount< getEntitiesDimension() >();
         BufferType buffer( 3 * entitiesCount );
         IndexType k = 0;
         for( IndexType i = 0; i < entitiesCount; i++ ) {
            const VectorType vector = getElement( i );
            static_assert( getVectorDimension() <= 3, "VTK formats support only up to 3D vector fields." );
            for( int j = 0; j < 3; j++ )
               buffer[ k++ ] = ( j < vector.getSize() ? vector[ j ] : 0 );
         }

         if( format == "vtk" ) {
            Meshes::Writers::VTKWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
         else if( format == "vtu" ) {
            Meshes::Writers::VTUWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
         else if( format == "vti" ) {
            Meshes::Writers::VTIWriter< Mesh > writer( file );
            writer.template writeEntities< getEntitiesDimension() >( *getMeshPointer() );
            if( getEntitiesDimension() == 0 )
               writer.writePointData( buffer, functionName, 3 );
            else
               writer.writeCellData( buffer, functionName, 3 );
         }
      }
      else if( format == "gnuplot" || format == "gplt" || format == "plt" )
         return VectorFieldGnuplotWriter< VectorField >::write( *this, file );
      else {
         std::cerr << "Unknown output format: " << format << std::endl;
         return false;
      }
      return true;
   }

protected:
   Containers::StaticArray< Size, FunctionPointer > vectorField;
};

template< int Dimension, typename Function >
std::ostream&
operator<<( std::ostream& str, const VectorField< Dimension, Function >& f )
{
   for( int i = 0; i < Dimension; i++ ) {
      str << "[ " << f[ i ] << " ]";
      if( i < Dimension - 1 )
         str << ", ";
   }
   return str;
}

}  // namespace Functions
}  // namespace noa::TNL
