#ifndef EulerVelXGetter_H
#define EulerVelXGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerVelXGetter
: public Functions::Domain< Mesh::getMeshDimensions(), Functions::MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      EulerVelXGetter( const MeshFunctionType& rho,
                       const MeshFunctionType& rhoVel)
      : rho( rho ), rhoVel( rhoVel )
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
          if (this->rho[ idx ]==0) return 0; else return this->rhoVel[ idx ] / this->rho[ idx ];
      }

      
   protected:
      
      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVel;

};

} // namespace TNL

#endif	/* EulerVelXGetter_H */
