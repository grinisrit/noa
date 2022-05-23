#pragma once 

//#include <TNL/Meshes/Grid.h>
//#include <TNL/Functions/MeshFunctionView.h>
//#include <TNL/Devices/Cuda.h>

using namespace TNL;

template< typename Mesh >
class tnlDirectEikonalMethodsBase
{   
};

template< typename Real,
        typename Device,
        typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >
{
  public:
    
    typedef Meshes::Grid< 1, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DevcieType;
    typedef Index IndexType;
    typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
    typedef Functions::MeshFunctionView< MeshType, 1, bool > InterfaceMapType;
    using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
    using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;
    
    void initInterface( const MeshFunctionPointer& input,
            MeshFunctionPointer& output,
            InterfaceMapPointer& interfaceMap );
    
    template< typename MeshEntity >
    __cuda_callable__ void updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            const RealType velocity = 1.0  );
    
    __cuda_callable__ bool updateCell( volatile Real sArray[18],
            int thri, const Real h,
            const Real velocity = 1.0 );
};


template< typename Real,
        typename Device,
        typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >
{
  public:
    typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DevcieType;
    typedef Index IndexType;
    typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
    typedef Functions::MeshFunctionView< MeshType, 2, bool > InterfaceMapType;
    typedef TNL::Containers::Array< int, Device, IndexType > ArrayContainer;
    using ArrayContainerView = typename ArrayContainer::ViewType;
    typedef Containers::StaticVector< 2, Index > StaticVector;
    
    using MeshPointer = Pointers::SharedPointer<  MeshType >;
    using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
    using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;
    
    // CALLER FOR HOST AND CUDA
    void initInterface( const MeshFunctionPointer& input,
            MeshFunctionPointer& output,
            InterfaceMapPointer& interfaceMap,
            StaticVector vLower, StaticVector vUpper );
        
    // FOR HOST
    template< typename MeshEntity >
    __cuda_callable__ bool updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            const RealType velocity = 1.0 );
    
    // FOR CUDA
    template< int sizeSArray >
    __cuda_callable__ bool updateCell( volatile RealType *sArray,
            int thri, int thrj, const RealType hx, const RealType hy,
            const RealType velocity = 1.0 );
        
// FOR OPENMP WILL BE REMOVED
    void getNeighbours( ArrayContainerView BlockIterHost, int numBlockX, int numBlockY  );
        
    template< int sizeSArray >
    void updateBlocks( InterfaceMapType interfaceMap,
            MeshFunctionType aux,
            MeshFunctionType helpFunc,
            ArrayContainerView BlockIterHost, int numThreadsPerBlock );
    
  protected:
    
   __cuda_callable__ RealType getNewValue( RealType valuesAndSteps[],
           const RealType originalValue, const RealType v );
};

template< typename Real,
        typename Device,
        typename Index >
class tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >
{
  public:
    typedef Meshes::Grid< 3, Real, Device, Index > MeshType;
    typedef Real RealType;
    typedef Device DevcieType;
    typedef Index IndexType;
    typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
    typedef Functions::MeshFunctionView< MeshType, 3, bool > InterfaceMapType;
    typedef TNL::Containers::Array< int, Device, IndexType > ArrayContainer;
    using ArrayContainerView = typename ArrayContainer::ViewType;
    typedef Containers::StaticVector< 3, Index > StaticVector;
    using MeshFunctionPointer = Pointers::SharedPointer< MeshFunctionType >;
    using InterfaceMapPointer = Pointers::SharedPointer< InterfaceMapType >;      
    
     // CALLER FOR HOST AND CUDA
    void initInterface( const MeshFunctionPointer& input,
            MeshFunctionPointer& output,
            InterfaceMapPointer& interfaceMap,
            StaticVector vLower, StaticVector vUpper );
    
    // FOR HOST
    template< typename MeshEntity >
    __cuda_callable__ bool updateCell( MeshFunctionType& u,
            const MeshEntity& cell,
            const RealType velocity = 1.0);
    
    // FOR CUDA
    template< int sizeSArray >
    __cuda_callable__ bool updateCell( volatile Real *sArray,
            int thri, int thrj, int thrk, const RealType hx, const RealType hy, const RealType hz,
            const RealType velocity = 1.0 );
    
    // OPENMP WILL BE REMOVED
    void getNeighbours( ArrayContainerView BlockIterHost, int numBlockX, int numBlockY, int numBlockZ );
    
    template< int sizeSArray >
    void updateBlocks( const InterfaceMapType interfaceMap,
            const MeshFunctionType aux,
            MeshFunctionType& helpFunc,
            ArrayContainer BlockIterHost, int numThreadsPerBlock );
    
  protected:
    
    __cuda_callable__ RealType getNewValue( RealType valuesAndSteps[],
           const RealType originalValue, const RealType v );
};

template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] );

#ifdef HAVE_CUDA
// 1D
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap  );

template < typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > > ptr,
        const Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap,
        Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& aux,
        bool *BlockIterDevice );




// 2D
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Containers::StaticVector< 2, Index > vecLowerOverlas,
        const Containers::StaticVector< 2, Index > vecUpperOerlaps );

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
        const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicator,
        const Containers::StaticVector< 2, Index > vecLowerOverlaps, 
        const Containers::StaticVector< 2, Index > vecUpperOverlaps, int oddEvenBlock =0);

template < typename Index >
__global__ void GetNeighbours( const TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicator,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicatorHelp, int numBlockX, int numBlockY );



// 3D
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        Containers::StaticVector< 3, Index > vecLowerOverlaps, Containers::StaticVector< 3, Index > vecUpperOverlaps );

template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
        const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        Containers::StaticVector< 3, Index > vecLowerOverlaps, Containers::StaticVector< 3, Index > vecUpperOverlaps );

template < typename Index >
__global__ void GetNeighbours( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom,
        int numBlockX, int numBlockY, int numBlockZ );
#endif

#include "tnlDirectEikonalMethodBase1D_impl.h"
#include "tnlDirectEikonalMethodBase2D_impl.h"
#include "tnlDirectEikonalMethodBase3D_impl.h"
