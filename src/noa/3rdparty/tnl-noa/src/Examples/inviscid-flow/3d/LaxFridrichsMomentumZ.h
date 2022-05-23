#ifndef LaxFridrichsMomentumZ_H
#define LaxFridrichsMomentumZ_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsMomentumZ
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumZ< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      Real tau;
      MeshFunctionType velocityX;
      MeshFunctionType velocityY;
      MeshFunctionType velocityZ;
      MeshFunctionType pressure;

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityX(MeshFunctionType& velocityX)
      {
          this->velocityX.bind(velocityX);
      };

      void setVelocityY(MeshFunctionType& velocityY)
      {
          this->velocityY.bind(velocityY);
      };

      void setVelocityZ(MeshFunctionType& velocityZ)
      {
          this->velocityZ.bind(velocityZ);
      };

      void setPressure(MeshFunctionType& pressure)
      {
          this->pressure.bind(pressure);
      };

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumZ< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      Real tau;
      MeshFunctionType velocityX;
      MeshFunctionType velocityY;
      MeshFunctionType velocityZ;
      MeshFunctionType pressure;

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityX(MeshFunctionType& velocityX)
      {
          this->velocityX.bind(velocityX);
      };

      void setVelocityY(MeshFunctionType& velocityY)
      {
          this->velocityY.bind(velocityY);
      };

      void setVelocityZ(MeshFunctionType& velocityZ)
      {
          this->velocityZ.bind(velocityZ);
      };

      void setPressure(MeshFunctionType& pressure)
      {
          this->pressure.bind(pressure);
      };

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumZ< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      enum { Dimensions = MeshType::getMeshDimensions() };

      Real tau;
      MeshFunctionType velocityX;
      MeshFunctionType velocityY;
      MeshFunctionType velocityZ;
      MeshFunctionType pressure;

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityX(MeshFunctionType& velocityX)
      {
          this->velocityX.bind(velocityX);
      };

      void setVelocityY(MeshFunctionType& velocityY)
      {
          this->velocityY.bind(velocityY);
      };

      void setVelocityZ(MeshFunctionType& velocityZ)
      {
          this->velocityZ.bind(velocityZ);
      };

      void setPressure(MeshFunctionType& pressure)
      {
          this->pressure.bind(pressure);
      };

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;

      template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;
};

} // namespace TNL

#include "LaxFridrichsMomentumZ_impl.h"

#endif	/* LaxFridrichsMomentumZ_H */
