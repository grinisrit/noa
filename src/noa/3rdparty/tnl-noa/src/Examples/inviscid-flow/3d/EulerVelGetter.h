#ifndef EulerVelGetter_H
#define EulerVelGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelGetter
: public Functions::Domain< Mesh::getMeshDimensions(), Functions::MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      EulerVelGetter( const MeshFunctionType& rho,
                      const MeshFunctionType& rhoVelX,
                      const MeshFunctionType& rhoVelY,
                      const MeshFunctionType& rhoVelZ)
      : rho( rho ), rhoVelX( rhoVelX ), rhoVelY( rhoVelY ), rhoVelZ( rhoVelZ )
      {}

      template< typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         return this->operator[]( entity.getIndex() );
      }
      
      __cuda_callable__
      Real operator[]( const IndexType& idx ) const
      {
         if (this->rho[ idx ]==0) return 0; else return std::sqrt( std::pow( this->rhoVelX[ idx ] / this->rho[ idx ], 2) + std::pow( this->rhoVelY[ idx ] / this->rho[ idx ], 2) + std::pow( this->rhoVelZ[ idx ] / this->rho[ idx ], 2) ) ;
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;

      const MeshFunctionType& rhoVelY;

      const MeshFunctionType& rhoVelZ;

};

} // namespace TNL

#endif	/* EulerVelGetter_H */
