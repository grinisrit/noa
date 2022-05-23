#pragma once

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap  )
{
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef HAVE_CUDA
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 16 );
    int numBlocksX = Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    dim3 blockSize( cudaBlockSize );
    dim3 gridSize( numBlocksX );
    Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
    CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >() );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    const MeshType& mesh = _input->getMesh();
    typedef typename MeshType::Cell Cell;
    const MeshFunctionType& input = _input.getData();
    MeshFunctionType& output = _output.modifyData();
    InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
    Cell cell( mesh );
    for( cell.getCoordinates().x() = 0;
            cell.getCoordinates().x() < mesh.getDimensions().x();
            cell.getCoordinates().x() ++ )
    {
      cell.refresh();
      output[ cell.getIndex() ] =
              input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                -std::numeric_limits< RealType >::max();
      interfaceMap[ cell.getIndex() ] = false;
    }
    
    
    const RealType& h = mesh.getSpaceSteps().x();
    for( cell.getCoordinates().x() = 0;
            cell.getCoordinates().x() < mesh.getDimensions().x() - 1;
            cell.getCoordinates().x() ++ )
    {
      cell.refresh();
      const RealType& c = input( cell );      
      if( ! cell.isBoundaryEntity()  )
      {
        const auto& neighbors = cell.getNeighborEntities();
        Real pom = 0;
        //const IndexType& c = cell.getIndex();
        const IndexType e = neighbors.template getEntityIndex<  1 >();
        if( c * input[ e ] <= 0 )
        {
          pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
          if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) )
            output[ cell.getIndex() ] = pom;
          
          pom = pom - TNL::sign( c )*h; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
          if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
            output[ e ] = pom; 
          
          interfaceMap[ cell.getIndex() ] = true;
          interfaceMap[ e ] = true;
        }
      }
    }
  }
}

template< typename Real,
        typename Device,
        typename Index >
template< typename MeshEntity >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell, 
        const RealType v )
{
  const auto& neighborEntities = cell.template getNeighborEntities< 1 >();
  const MeshType& mesh = cell.getMesh();
  const RealType& h = mesh.getSpaceSteps().x();
  const RealType value = u( cell );
  RealType a, tmp = std::numeric_limits< RealType >::max();
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1 >() ],
            u[ neighborEntities.template getEntityIndex<  1 >() ] );
  }
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() )
    return;
  
  tmp = a + TNL::sign( value ) * h/v;
  
  u[ cell.getIndex() ] = argAbsMin( value, tmp );
}

template< typename Real,
        typename Device,
        typename Index >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 1, Real, Device, Index > >::
updateCell( volatile Real sArray[18], int thri, const Real h, const Real v )
{
  const RealType value = sArray[ thri ];
  RealType a, tmp = std::numeric_limits< RealType >::max();
  
  a = TNL::argAbsMin( sArray[ thri+1 ],
          sArray[ thri-1 ] );
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() )
    return false;
  
  tmp = a + TNL::sign( value ) * h/v;
  
  
  sArray[ thri ] = argAbsMin( value, tmp );
  
  tmp = value - sArray[ thri ];
  if ( fabs( tmp ) >  0.001*h )
    return true;
  else
    return false;
}

#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 1, Real, Device, Index >, 1, bool >& interfaceMap )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  const Meshes::Grid< 1, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x()  )
  {
    typedef typename Meshes::Grid< 1, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    
    const Real& h = mesh.getSpaceSteps().x();
    cell.refresh();
    const Real& c = input( cell );
    if( ! cell.isBoundaryEntity()  )
    {
      auto neighbors = cell.getNeighborEntities();
      Real pom = 0;
      const Index e = neighbors.template getEntityIndex< 1 >();
      const Index w = neighbors.template getEntityIndex< -1 >();
      if( c * input[ e ] <= 0 )
      {
        pom = TNL::sign( c )*( h * c )/( c - input[ e ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
          output[ cind ] = pom;                       
        
        interfaceMap[ cind ] = true;
      }
      if( c * input[ w ] <= 0 )
      {
        pom = TNL::sign( c )*( h * c )/( c - input[ w ]);
        if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
          output[ cind ] = pom;
        
        interfaceMap[ cind ] = true;
      }
    }
  }
  
}
#endif


