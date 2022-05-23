#pragma once

//#include <TNL/Meshes/Grid.h>
//#include <TNL/Functions/Analytic/Constant.h>
//#include <TNL/Pointers/SharedPointer.h>
#include "tnlDirectEikonalMethodsBase.h"
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>


template< typename Mesh,
          typename Anisotropy = Functions::Analytic::Constant< Mesh::getMeshDimension(), typename Mesh::RealType > >
class FastSweepingMethod
{
};

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 1, Real, Device, Index >, Anisotropy >
: public tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >
{
  //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );

  public:

    typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef Anisotropy AnisotropyType;
    typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > > BaseType;
    using MeshPointer = Pointers::SharedPointer<  MeshType >;
    using AnisotropyPointer = Pointers::SharedPointer< AnisotropyType, DeviceType >;


    using typename BaseType::InterfaceMapType;
    using typename BaseType::MeshFunctionType;
    using typename BaseType::InterfaceMapPointer;
    using typename BaseType::MeshFunctionPointer;


    FastSweepingMethod();

    const IndexType& getMaxIterations() const;

    void setMaxIterations( const IndexType& maxIterations );

    void solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
            const MeshPointer& mesh,
            MeshFunctionPointer& Aux,
            const AnisotropyPointer& anisotropy,
            MeshFunctionPointer& u );


    protected:

      const IndexType maxIterations;
};

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >
: public tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >
{
  //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );

  public:

    typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef Anisotropy AnisotropyType;
    typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > BaseType;
    typedef Containers::StaticVector< 2, Index > StaticVector;

    using MeshPointer = Pointers::SharedPointer<  MeshType >;
    using AnisotropyPointer = Pointers::SharedPointer< AnisotropyType, DeviceType >;

    using typename BaseType::InterfaceMapType;
    using typename BaseType::MeshFunctionType;
    using typename BaseType::InterfaceMapPointer;
    using typename BaseType::MeshFunctionPointer;
    using typename BaseType::ArrayContainer;

    FastSweepingMethod();

    const IndexType& getMaxIterations() const;

    void setMaxIterations( const IndexType& maxIterations );

    void solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
            const MeshPointer& mesh,
            MeshFunctionPointer& Aux,
            const AnisotropyPointer& anisotropy,
            const MeshFunctionPointer& u );

    protected:

      using DistributedMeshSynchronizerType = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Meshes::DistributedMeshes::DistributedMesh< typename MeshFunctionType::MeshType > >;
      DistributedMeshSynchronizerType synchronizer;

      const IndexType maxIterations;

      bool goThroughSweep( const StaticVector boundsFrom, const StaticVector boundsTo,
              MeshFunctionType& aux, const InterfaceMapType& interfaceMap,
              const AnisotropyPointer& anisotropy );

      void getInfoFromNeighbours( int& calculated, int& calculateAgain,
                                  const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh );
};

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
class FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >
: public tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >
{
  //static_assert(  std::is_same< Device, TNL::Devices::Host >::value, "The fast sweeping method works only on CPU." );

  public:

    typedef Meshes::Grid< 3, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef Anisotropy AnisotropyType;
    typedef tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > BaseType;
    typedef Containers::StaticVector< 3, Index > StaticVector;

    using MeshPointer = Pointers::SharedPointer<  MeshType >;
    using AnisotropyPointer = Pointers::SharedPointer< AnisotropyType, DeviceType >;

    using typename BaseType::InterfaceMapType;
    using typename BaseType::MeshFunctionType;
    using typename BaseType::InterfaceMapPointer;
    using typename BaseType::MeshFunctionPointer;
    using typename BaseType::ArrayContainer;


    FastSweepingMethod();

    const IndexType& getMaxIterations() const;

    void setMaxIterations( const IndexType& maxIterations );

    void solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
            const MeshPointer& mesh,
            MeshFunctionPointer& Aux,
            const AnisotropyPointer& anisotropy,
            MeshFunctionPointer& u );


    protected:

      using DistributedMeshSynchronizerType = Meshes::DistributedMeshes::DistributedMeshSynchronizer< Meshes::DistributedMeshes::DistributedMesh< typename MeshFunctionType::MeshType > >;
      DistributedMeshSynchronizerType synchronizer;

      const IndexType maxIterations;

      bool goThroughSweep( const StaticVector boundsFrom, const StaticVector boundsTo,
              MeshFunctionType& aux, const InterfaceMapType& interfaceMap,
              const AnisotropyPointer& anisotropy );

      void getInfoFromNeighbours( int& calculated, int& calculateAgain,
                                  const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh );
};


#include "tnlFastSweepingMethod1D_impl.h"
#include "tnlFastSweepingMethod2D_impl.h"
#include "tnlFastSweepingMethod3D_impl.h"
