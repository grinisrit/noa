#ifndef EulerPressureGetter_H
#define EulerPressureGetter_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Domain.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class EulerPressureGetter
: public Functions::Domain< Mesh::getMeshDimensions(), Functions::MeshDomain >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      EulerPressureGetter( const MeshFunctionType& rho,
                           const MeshFunctionType& rhoVelX,
                           const MeshFunctionType& rhoVelY,
                           const MeshFunctionType& rhoVelZ,
                           const MeshFunctionType& energy,
                           const RealType& gamma )
      : rho( rho ), rhoVelX( rhoVelX ), rhoVelY( rhoVelY ), rhoVelZ( rhoVelZ ), energy( energy ), gamma( gamma )
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
/*         if (this->rho[ idx ]==0) return 0; 
         else return ( this->gamma - 1.0 ) * ( this->energy[ idx ] - 0.5 * this->rho[ idx ] * 
         ( std::pow(this->rhoVelX[ idx ] / this->rho[ idx ],2) + std::pow(this->rhoVelY[ idx ] / this->rho[ idx ],2) + std::pow(this->rhoVelZ[ idx ] / this->rho[ idx ],2) );
*/       return ( this->gamma - 1.0 ) * ( this->energy[ idx ] * this->rho[ idx ] );
      }

      
   protected:

      const MeshFunctionType& rho;
      
      const MeshFunctionType& rhoVelX;
      
      const MeshFunctionType& rhoVelY;

      const MeshFunctionType& rhoVelZ;

      const MeshFunctionType& energy;

      RealType gamma;

};

} //namespace TNL

#endif	/* EulerPressureGetter_H */
