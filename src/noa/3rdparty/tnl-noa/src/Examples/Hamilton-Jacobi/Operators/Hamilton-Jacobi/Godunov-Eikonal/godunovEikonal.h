#pragma once 

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <TNL/Containers/Vector.h>
#include <core/mfilename.h>
#include <mesh/tnlGrid.h>


template< typename Mesh,
		  typename Real,
		  typename Index >
class godunovEikonalScheme
{
};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class godunovEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 1, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;



	static String getType();

	RealType positivePart(const RealType arg) const;

	RealType negativePart(const RealType arg) const;

	RealType sign(const RealType x, const RealType eps) const;
   
   template< typename PreimageFunction, typename MeshEntity >
   __cuda_callable__
    Real operator()( const PreimageFunction& u,
                     const MeshEntity& entity,
                     const RealType& time = 0.0 ) const;

	bool init( const Config::ParameterContainer& parameters );


protected:

	MeshType originalMesh;

	DofVectorType dofVector;

	RealType h;

	RealType epsilon;


};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class godunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{

   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::CoordinatesType CoordinatesType;

      static String getType();

      RealType positivePart(const RealType arg) const;

      RealType negativePart(const RealType arg) const;

      RealType sign(const RealType x, const Real eps) const;

      template< typename PreimageFunction, typename MeshEntity >
      __cuda_callable__
       Real operator()( const PreimageFunction& u,
                        const MeshEntity& entity,
                        const RealType& time = 0.0 ) const;

      bool init( const Config::ParameterContainer& parameters );

   protected:

      MeshType originalMesh;

      DofVectorType dofVector;

      RealType hx;
      RealType hy;

      RealType epsilon;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class godunovEikonalScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{

   public:
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlGrid< 3, Real, Device, Index > MeshType;
      typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::CoordinatesType CoordinatesType;


      static String getType();

      RealType positivePart(const RealType arg) const;

      RealType negativePart(const RealType arg) const;

      RealType sign(const RealType x, const Real eps) const;

      template< typename PreimageFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const PreimageFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const;


      bool init( const Config::ParameterContainer& parameters );


   protected:

      MeshType originalMesh;

      DofVectorType dofVector;

      RealType hx;
      RealType hy;
      RealType hz;

      RealType epsilon;

};

#include <operators/hamilton-jacobi/godunov-eikonal/godunovEikonal1D_impl.h>
#include <operators/hamilton-jacobi/godunov-eikonal/godunovEikonal2D_impl.h>
#include <operators/hamilton-jacobi/godunov-eikonal/godunovEikonal3D_impl.h>


