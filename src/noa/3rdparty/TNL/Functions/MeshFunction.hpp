// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Pointers/DevicePointer.h>
#include <noa/3rdparty/TNL/Functions/MeshFunction.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionEvaluator.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionNormGetter.h>
#include <noa/3rdparty/TNL/Functions/MeshFunctionIO.h>

namespace noa::TNL {
namespace Functions {

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction()
{
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshPointer& meshPointer )
{
   this->meshPointer = meshPointer;
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshFunction& meshFunction )
{
   this->meshPointer = meshFunction.meshPointer;
   this->data = meshFunction.getData();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
setMesh( const MeshPointer& meshPointer )
{
   this->meshPointer = meshPointer;
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
 template< typename Device >
__cuda_callable__
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::MeshType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMesh() const
{
   return this->meshPointer.template getData< Device >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMeshPointer() const
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunction< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMeshPointer()
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunction< Mesh, MeshEntityDimension, Real >::IndexType
MeshFunction< Mesh, MeshEntityDimension, Real >::
getDofs( const MeshPointer& meshPointer )
{
   return meshPointer->template getEntitiesCount< getEntitiesDimension() >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::VectorType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename MeshFunction< Mesh, MeshEntityDimension, Real >::VectorType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
refresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
deepRefresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
const typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator = ( const MeshFunction& f )
{
   this->setMesh( f.getMeshPointer() );
   this->getData() = f.getData();
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator = ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator += ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) 1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator -= ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getLpNorm( const RealType& p ) const
{
   return MeshFunctionNormGetter< Mesh >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMaxNorm() const
{
   return max( abs( this->data ) );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
operator << ( std::ostream& str, const MeshFunction< Mesh, MeshEntityDimension, Real >& f )
{
   str << f.getData();
   return str;
}

} // namespace Functions
} // namespace noa::TNL
