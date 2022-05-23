#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../typedefs.h"
#include "StaticVector.h"
#include "mesh_getters.h"

#include <TNL/String.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>

#include <type_traits>

template< typename MeshEntity,
          int Superdimension,
          typename Scope,
          std::enable_if_t< Superdimension <= MeshEntity::MeshType::getMeshDimension(), bool > = true >
//                                           && MeshEntity::template SuperentityTraits< Superdimension >::storageEnabled >::type >
void export_getSuperentityIndex( Scope & m )
{
    m.def("getSuperentityIndex", []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                                        return entity.template getSuperentityIndex< Superdimension >( i );
                                    }
            );
}

template< typename MeshEntity,
          int Superdimension,
          typename Scope,
          std::enable_if_t< ! ( Superdimension <= MeshEntity::MeshType::getMeshDimension() ), bool > = true >
void export_getSuperentityIndex( Scope & )
{
}


template< typename MeshEntity,
          int Subdimension,
          typename Scope,
          std::enable_if_t< Subdimension <= MeshEntity::MeshType::getMeshDimension()
                            && (Subdimension < MeshEntity::getEntityDimension()), bool > = true >
void export_getSubentityIndex( Scope & m, const char* name )
{
    m.def(name, []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                    return entity.template getSubentityIndex< Subdimension >( i );
                }
            );
}

template< typename MeshEntity,
          int Subdimension,
          typename Scope,
          std::enable_if_t< ! ( Subdimension <= MeshEntity::MeshType::getMeshDimension()
                                && (Subdimension < MeshEntity::getEntityDimension())
                              ), bool > = true >
void export_getSubentityIndex( Scope &, const char* )
{
}


template< typename MeshEntity,
          typename Scope,
          std::enable_if_t< MeshEntity::getEntityDimension() == 0, bool > = true >
void export_getPoint( Scope & scope )
{
    scope.def("getPoint", []( const MeshEntity& entity ) {
                            return entity.getPoint();
                        }
            );
}

template< typename MeshEntity,
          typename Scope,
          std::enable_if_t< MeshEntity::getEntityDimension() != 0, bool > = true >
void export_getPoint( Scope & )
{
}


template< typename MeshEntity, typename Scope >
void export_MeshEntity( Scope & scope, const char* name )
{
    auto entity = py::class_< MeshEntity >( scope, name )
//        .def(py::init<>())
//        .def(py::init<typename MeshEntity::MeshType, typename MeshEntity::GlobalIndexType>())
        .def_static("getEntityDimension", &MeshEntity::getEntityDimension)
        .def("getIndex", &MeshEntity::getIndex)
        .def("getTag", &MeshEntity::getTag)
        // TODO
    ;

    export_getSuperentityIndex< MeshEntity, MeshEntity::getEntityDimension() + 1 >( entity );
    export_getSubentityIndex< MeshEntity, 0 >( entity, "getSubvertexIndex" );
    export_getPoint< MeshEntity >( entity );
}

template< typename Mesh >
void export_Mesh( py::module & m, const char* name )
{
    auto mesh = py::class_< Mesh >( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Mesh::getMeshDimension)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Mesh, typename Mesh::Cell >)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Mesh, typename Mesh::Face >)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Mesh, typename Mesh::Vertex >)
        .def("getGhostEntitiesCount", &mesh_getGhostEntitiesCount< Mesh, typename Mesh::Cell >)
        .def("getGhostEntitiesCount", &mesh_getGhostEntitiesCount< Mesh, typename Mesh::Face >)
        .def("getGhostEntitiesCount", &mesh_getGhostEntitiesCount< Mesh, typename Mesh::Vertex >)
        .def("getGhostEntitiesOffset", &mesh_getGhostEntitiesOffset< Mesh, typename Mesh::Cell >)
        .def("getGhostEntitiesOffset", &mesh_getGhostEntitiesOffset< Mesh, typename Mesh::Face >)
        .def("getGhostEntitiesOffset", &mesh_getGhostEntitiesOffset< Mesh, typename Mesh::Vertex >)
        // NOTE: if combined into getEntity, the return type would depend on the runtime parameter (entity)
        .def("getCell", &Mesh::template getEntity<typename Mesh::Cell>)
        .def("getFace", &Mesh::template getEntity<typename Mesh::Face>)
        .def("getVertex", &Mesh::template getEntity<typename Mesh::Vertex>)
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Cell& cell ){ return getEntityCenter( mesh, cell ); } )
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Face& face ){ return getEntityCenter( mesh, face ); } )
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){ return getEntityCenter( mesh, vertex ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Cell& cell ){ return getEntityMeasure( mesh, cell ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Face& face ){ return getEntityMeasure( mesh, face ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){ return getEntityMeasure( mesh, vertex ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Cell& cell ){
                                       return mesh.template isBoundaryEntity< Mesh::Cell::getEntityDimension() >( cell.getIndex() ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Face& face ){
                                       return mesh.template isBoundaryEntity< Mesh::Face::getEntityDimension() >( face.getIndex() ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){
                                        return mesh.template isBoundaryEntity< Mesh::Vertex::getEntityDimension() >( vertex.getIndex() ); } )
        .def("isGhostEntity", []( const Mesh& mesh, const typename Mesh::Cell& cell ){
                                       return mesh.template isGhostEntity< Mesh::Cell::getEntityDimension() >( cell.getIndex() ); } )
        .def("isGhostEntity", []( const Mesh& mesh, const typename Mesh::Face& face ){
                                       return mesh.template isGhostEntity< Mesh::Face::getEntityDimension() >( face.getIndex() ); } )
        .def("isGhostEntity", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){
                                        return mesh.template isGhostEntity< Mesh::Vertex::getEntityDimension() >( vertex.getIndex() ); } )
        // TODO: more?
    ;

    // nested types
    export_MeshEntity< typename Mesh::Cell >( mesh, "Cell" );
    export_MeshEntity< typename Mesh::Face >( mesh, "Face" );
    // avoid duplicate conversion if the type is the same
    if( ! std::is_same< typename Mesh::Face, typename Mesh::Vertex >::value )
        export_MeshEntity< typename Mesh::Vertex >( mesh, "Vertex" );
}
