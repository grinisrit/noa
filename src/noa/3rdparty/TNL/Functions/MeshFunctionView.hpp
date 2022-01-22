// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Pointers/DevicePointer.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionView.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionEvaluator.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionNormGetter.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionIO.h>

namespace noaTNL {
namespace Functions {

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView()
{
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshFunctionView& meshFunction )
{
   this->meshPointer = meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshPointer& meshPointer,
              Vector& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data.getSize(), meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}


template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshPointer& meshPointer,
              Pointers::SharedPointer<  Vector >& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< String >( prefix + "file", "Dataset for the mesh function." );
   config.addEntry< String >( prefix + "function-name", "Name of the mesh function in the input file.", "f" );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setMesh( meshPointer );
   const String fileName = parameters.getParameter< String >( prefix + "file" );
   const String functionName = parameters.getParameter< String >( prefix + "function-name" );
   return readMeshFunction( *this, functionName, fileName );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( MeshFunctionView& meshFunction )
{
   this->meshPointer = meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
   this->data.bind( data.getData() + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( data.getData() + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      Pointers::SharedPointer< Vector >& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
   static_assert( std::is_same< typename Vector::RealType, RealType >::value, "Cannot bind Vector with different Real type." );

   this->meshPointer = meshPointer;
   this->data.bind( *data + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setMesh( const MeshPointer& meshPointer )
{
   this->meshPointer = meshPointer;
   this->data.reset();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
 template< typename Device >
__cuda_callable__
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::MeshType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMesh() const
{
   return this->meshPointer.template getData< Device >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMeshPointer() const
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMeshPointer()
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::IndexType
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getDofs( const MeshPointer& meshPointer )
{
   return meshPointer->template getEntitiesCount< getEntitiesDimension() >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::VectorType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::VectorType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
refresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
deepRefresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getValue( const EntityType& meshEntity ) const
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data.getElement( meshEntity.getIndex() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setValue( const EntityType& meshEntity,
          const RealType& value )
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   this->data.setElement( meshEntity.getIndex(), value );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
__cuda_callable__
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time )
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
__cuda_callable__
const typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time ) const
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator = ( const MeshFunctionView& f )
{
   this->setMesh( f.getMeshPointer() );
   this->getData() = f.getData();
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator = ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator += ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) 1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator -= ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getLpNorm( const RealType& p ) const
{
   return MeshFunctionNormGetter< Mesh >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMaxNorm() const
{
   return max( abs( this->data ) );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
write( const std::string& functionName,
       const std::string& fileName,
       const std::string& fileFormat ) const
{
   return writeMeshFunction( *this, functionName, fileName, fileFormat );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
std::ostream&
operator << ( std::ostream& str, const MeshFunctionView< Mesh, MeshEntityDimension, Real >& f )
{
   str << f.getData();
   return str;
}

} // namespace Functions
} // namespace noaTNL
