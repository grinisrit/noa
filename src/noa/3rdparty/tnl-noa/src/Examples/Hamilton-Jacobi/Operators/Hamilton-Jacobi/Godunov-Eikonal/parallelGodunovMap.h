#ifndef PARALLELGODUNOVMAP_H_
#define PARALLELGODUNOVMAP_H_

#include <matrices/tnlCSRMatrix.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>
#include <TNL/Containers/Vector.h>
#include <core/mfilename.h>
#include <mesh/tnlGrid.h>
#include <core/tnlCuda.h>
#include <mesh/grids/tnlGridEntity.h>


template< typename Mesh,
		  typename Real,
		  typename Index >
class parallelGodunovMapScheme
{
};




template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class parallelGodunovMapScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 1, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;


#ifdef HAVE_CUDA
   __device__ __host__
#endif
	static String getType();
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	RealType positivePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	RealType negativePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	RealType sign(const RealType x, const RealType eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time,
                   const IndexType boundaryCondition) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	bool init( const Config::ParameterContainer& parameters );

   RealType epsilon;

protected:

	MeshType originalMesh;

	DofVectorType dofVector;

	RealType h;




};






template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class parallelGodunovMapScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 2, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	static String getType();
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType positivePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType negativePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType sign(const RealType x, const Real eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time,
                   const IndexType boundaryCondition,
                   const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities,
                   const Vector& map) const;

 #ifdef HAVE_CUDA
    __device__
 #endif
    Real getValueDev( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const RealType* u,
                   const RealType& time,
                   const IndexType boundaryCondition,
                   const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities,
  	  	  	       const RealType* map) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
	bool init( const Config::ParameterContainer& parameters );

   RealType epsilon;

protected:

 	MeshType originalMesh;

    DofVectorType dofVector;

    RealType hx, ihx;
    RealType hy, ihy;




};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class parallelGodunovMapScheme< tnlGrid< 3,MeshReal, Device, MeshIndex >, Real, Index >
{

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef tnlGrid< 3, Real, Device, Index > MeshType;
	typedef TNL::Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
	typedef typename MeshType::CoordinatesType CoordinatesType;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
	static String getType();
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType positivePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType negativePart(const RealType arg) const;
#ifdef HAVE_CUDA
   __device__ __host__
#endif
    RealType sign(const RealType x, const Real eps) const;

    template< typename Vector >
 #ifdef HAVE_CUDA
    __device__ __host__
 #endif
    Real getValue( const MeshType& mesh,
                   const IndexType cellIndex,
                   const CoordinatesType& coordinates,
                   const Vector& u,
                   const RealType& time,
                   const IndexType boundaryCondition,
  	  	  	       const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighborEntities  ) const;

#ifdef HAVE_CUDA
   __device__
#endif
   Real getValueDev( const MeshType& mesh,
                  const IndexType cellIndex,
                  const CoordinatesType& coordinates,
                  const RealType* u,
                  const RealType& time,
                  const IndexType boundaryCondition,
 	  	  	      const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighborEntities) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
    bool init( const Config::ParameterContainer& parameters );

   RealType epsilon;

protected:

 	MeshType originalMesh;

    DofVectorType dofVector;

    RealType hx, ihx;
    RealType hy, ihy;
    RealType hz, ihz;

};



//#include <operators/godunov-eikonal/parallelGodunovMap1D_impl.h>
#include <operators/hamilton-jacobi/godunov-eikonal/parallelGodunovMap2D_impl.h>
//#include <operators/godunov-eikonal/parallelGodunovMap3D_impl.h>


#endif /* PARALLELGODUNOVMAP_H_ */
