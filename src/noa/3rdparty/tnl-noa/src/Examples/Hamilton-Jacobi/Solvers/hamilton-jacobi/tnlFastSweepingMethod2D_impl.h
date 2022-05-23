#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/MPI/Wrappers.h>

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
       const MeshPointer& mesh,
        MeshFunctionPointer& Aux,
        const AnisotropyPointer& anisotropy,
        const MeshFunctionPointer& u )
{
  MeshFunctionPointer auxPtr;
  InterfaceMapPointer interfaceMapPtr;
  auxPtr->setMesh( mesh );
  interfaceMapPtr->setMesh( mesh );

  // Setting overlaps ( WITHOUT MPI SHOULD BE 0 )
  StaticVector vecLowerOverlaps = 0;
  StaticVector vecUpperOverlaps = 0;
  if( TNL::MPI::GetSize() > 1 )
  {
    //Distributed mesh for MPI overlaps (without MPI null pointer)
    vecLowerOverlaps = distributedMesh.getLowerOverlap();
    vecUpperOverlaps = distributedMesh.getUpperOverlap();
  }

  std::cout << "Initiating the interface cells ..." << std::endl;
  BaseType::initInterface( u, auxPtr, interfaceMapPtr, vecLowerOverlaps, vecUpperOverlaps );

  //auxPtr->write( "aux", "aux-ini.vti" );

  typename MeshType::Cell cell( *mesh );

  IndexType iteration( 0 );
  InterfaceMapType interfaceMap = *interfaceMapPtr;
  MeshFunctionType aux = *auxPtr;
  synchronizer.setDistributedGrid( &distributedMesh );
  synchronizer.synchronize( aux ); //synchronize initialized overlaps

  std::cout << "Calculating the values ..." << std::endl;
  while( iteration < this->maxIterations )
  {
    // calculatedBefore indicates weather we calculated in the last passage of the while cycle
    // calculatedBefore is same for all ranks
    // without MPI should be FALSE at the end of while cycle body
    int calculatedBefore = 1;

    // calculateMPIAgain indicates if the thread should calculate again in upcoming passage of while cycle
    // calculateMPIAgain is a value that can differ in every rank
    // without MPI should be FALSE at the end of while cycle body
    int calculateMPIAgain = 1;

    while( calculatedBefore )
    {
      calculatedBefore = 0;

      if( std::is_same< DeviceType, Devices::Host >::value && calculateMPIAgain ) // should we calculate in Host?
      {
        calculateMPIAgain = 0;

  /**--HERE-IS-PARALLEL-OMP-CODE--!!!WITHOUT MPI!!!--------------------**/
        /*
         int numThreadsPerBlock = -1;

         numThreadsPerBlock = ( mesh->getDimensions().x()/2 + (mesh->getDimensions().x() % 2 != 0 ? 1:0));
         //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);
         if( numThreadsPerBlock <= 16 )
         numThreadsPerBlock = 16;
         else if(numThreadsPerBlock <= 32 )
         numThreadsPerBlock = 32;
         else if(numThreadsPerBlock <= 64 )
         numThreadsPerBlock = 64;
         else if(numThreadsPerBlock <= 128 )
         numThreadsPerBlock = 128;
         else if(numThreadsPerBlock <= 256 )
         numThreadsPerBlock = 256;
         else if(numThreadsPerBlock <= 512 )
         numThreadsPerBlock = 512;
         else
         numThreadsPerBlock = 1024;
         //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);

         if( numThreadsPerBlock == -1 ){
         printf("Fail in setting numThreadsPerBlock.\n");
         break;
         }



         int numBlocksX = mesh->getDimensions().x() / numThreadsPerBlock + (mesh->getDimensions().x() % numThreadsPerBlock != 0 ? 1:0);
         int numBlocksY = mesh->getDimensions().y() / numThreadsPerBlock + (mesh->getDimensions().y() % numThreadsPerBlock != 0 ? 1:0);

         //std::cout << "numBlocksX = " << numBlocksX << std::endl;

         //Real **sArray = new Real*[numBlocksX*numBlocksY];
         //for( int i = 0; i < numBlocksX * numBlocksY; i++ )
         // sArray[ i ] = new Real [ (numThreadsPerBlock + 2)*(numThreadsPerBlock + 2)];

         ArrayContainer BlockIterHost;
         BlockIterHost.setSize( numBlocksX * numBlocksY );
         BlockIterHost.setValue( 1 );
         int IsCalculationDone = 1;

         MeshFunctionPointer helpFunc( mesh );
         MeshFunctionPointer helpFunc1( mesh );
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         //std::cout<< "Size = " << BlockIterHost.getSize() << std::endl;
         //for( int k = numBlocksX-1; k >-1; k-- ){
         // for( int l = 0; l < numBlocksY; l++ ){
         // std::cout<< BlockIterHost[ l*numBlocksX  + k ];
         // }
         // std::cout<<std::endl;
         // }
         // std::cout<<std::endl;
         unsigned int numWhile = 0;
         while( IsCalculationDone )
         {
         IsCalculationDone = 0;
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         switch ( numThreadsPerBlock ){
         case 16:
         this->template updateBlocks< 18 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 32:
         this->template updateBlocks< 34 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 64:
         this->template updateBlocks< 66 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 128:
         this->template updateBlocks< 130 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 256:
         this->template updateBlocks< 258 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         case 512:
         this->template updateBlocks< 514 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         default:
         this->template updateBlocks< 1028 >( interfaceMap, *auxPtr, *helpFunc, BlockIterHost, numThreadsPerBlock );
         }


         //Reduction
         for( int i = 0; i < BlockIterHost.getSize(); i++ ){
         if( IsCalculationDone == 0 ){
         IsCalculationDone = IsCalculationDone || BlockIterHost[ i ];
         //break;
         }
         }
         numWhile++;
         //std::cout <<"numWhile = "<< numWhile <<std::endl;

         // for( int j = numBlocksY-1; j>-1; j-- ){
         // for( int i = 0; i < numBlocksX; i++ )
         // std::cout << BlockIterHost[ j * numBlocksX + i ];
         // std::cout << std::endl;
         // }
         // std::cout << std::endl;

         this->getNeighbours( BlockIterHost, numBlocksX, numBlocksY );

         //std::cout<<std::endl;
         //String s( "aux-"+ std::to_string(numWhile) + ".vti");
         //aux.write( "aux", s );
         }
         if( numWhile == 1 ){
         auxPtr = helpFunc;
         }
         */
  /**-END-OF-OMP-PARALLEL------------------------------------------------**/


  // FSM FOR MPI and WITHOUT MPI
        StaticVector boundsFrom; StaticVector boundsTo;
    // UP and RIGHT
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        calculatedBefore = goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.write( "aux", "aux-1.vti" );

    // UP and LEFL
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = -1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.write( "aux", "aux-2.vti" );

    // DOWN and RIGHT
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );
        //aux.write( "aux", "aux-3.vti" );

    // DOWN and LEFT
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

      }
      if( std::is_same< DeviceType, Devices::Cuda >::value && calculateMPIAgain ) // should we calculate on CUDA?
      {
        calculateMPIAgain = 0;

#ifdef HAVE_CUDA
        TNL_CHECK_CUDA_DEVICE;
        // Maximum cudaBlockSite is 32. Because of maximum num. of threads in kernel.
        // IF YOU CHANGE THIS, YOU NEED TO CHANGE THE TEMPLATE PARAMETER IN CudaUpdateCellCaller (The Number + 2)
        const int cudaBlockSize( 16 );

        // Setting number of threads and blocks for kernel
        int numBlocksX = Cuda::getNumberOfBlocks( mesh->getDimensions().x() - vecLowerOverlaps[0] - vecUpperOverlaps[0], cudaBlockSize );
        int numBlocksY = Cuda::getNumberOfBlocks( mesh->getDimensions().y() - vecLowerOverlaps[1] - vecUpperOverlaps[1], cudaBlockSize );
        dim3 blockSize( cudaBlockSize, cudaBlockSize );
        dim3 gridSize( numBlocksX, numBlocksY );

        // Need for calling functions from kernel
        BaseType ptr;

        // True if we should calculate again.
        int calculateCudaBlocksAgain = 1;

        // Array that identifies which blocks should be calculated.
        // All blocks should calculate in first passage ( setValue(1) )
        TNL::Containers::Array< int, Devices::Cuda, IndexType > blockCalculationIndicator( numBlocksX * numBlocksY );
        blockCalculationIndicator.setValue( 1 );
        TNL_CHECK_CUDA_DEVICE;

        // Array into which we identify the neighbours and then copy it into blockCalculationIndicator
        TNL::Containers::Array< int, Devices::Cuda, IndexType > blockCalculationIndicatorHelp(numBlocksX * numBlocksY );
        blockCalculationIndicatorHelp.setValue( 0 );

        // number of Blocks for kernel that calculates neighbours.
        int nBlocksNeigh = ( numBlocksX * numBlocksY )/1024 + ((( numBlocksX * numBlocksY )%1024 != 0) ? 1:0);

        // Helping meshFunction that switches with AuxPtr in every calculation of CudaUpdateCellCaller<<<>>>()
        Containers::Vector< RealType, DeviceType, IndexType > helpVec;
        helpVec.setLike( auxPtr.template getData().getData() );
        MeshFunctionPointer helpFunc;
        helpFunc->bind( mesh, helpVec );
        helpFunc.template modifyData() = auxPtr.template getData();

        // number of iterations of while calculateCudaBlocksAgain
        int numIter = 0;

        //int oddEvenBlock = 0;
        while( calculateCudaBlocksAgain )
        {
  /** HERE IS CHESS METHOD (NO MPI) **/

          /*
           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           auxPtr.template getData< Device>(),
           helpFunc.template modifyData< Device>(),
           blockCalculationIndicator, vecLowerOverlaps, vecUpperOverlaps,
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;

           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;

           CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr,
           interfaceMapPtr.template getData< Device >(),
           helpFunc.template getData< Device>(),
           auxPtr.template modifyData< Device>(),
           blockCalculationIndicator, vecLowerOverlaps, vecUpperOverlaps,
           oddEvenBlock );
           cudaDeviceSynchronize();
           TNL_CHECK_CUDA_DEVICE;

           oddEvenBlock= (oddEvenBlock == 0) ? 1: 0;

           calculateCudaBlocksAgain = Algorithms::contains( blockCalculationIndicator, 1);
          */
  /**------------------------------------------------------------------------------------------------*/


  /** HERE IS FIM FOR MPI AND WITHOUT MPI **/
          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
          CudaUpdateCellCaller<18><<< gridSize, blockSize >>>( ptr, interfaceMapPtr.template getData< Device >(),
                  auxPtr.template getData< Device>(), helpFunc.template modifyData< Device>(),
                  blockCalculationIndicator.getView(), vecLowerOverlaps, vecUpperOverlaps );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;

          // Switching helpFunc and auxPtr.
          auxPtr.swap( helpFunc );

          // Getting blocks that should calculate in next passage. These blocks are neighbours of those that were calculated now.
          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
          GetNeighbours<<< nBlocksNeigh, 1024 >>>( blockCalculationIndicator.getView(), blockCalculationIndicatorHelp.getView(), numBlocksX, numBlocksY );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          blockCalculationIndicator = blockCalculationIndicatorHelp;
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;

          // "Parallel reduction" to see if we should calculate again calculateCudaBlocksAgain
          calculateCudaBlocksAgain = Algorithms::contains( blockCalculationIndicator, 1);

          // When we change something then we should caclucate again in the next passage of MPI ( calculated = true )
         if( calculateCudaBlocksAgain ){
            calculatedBefore = 1;
          }

/**-----------------------------------------------------------------------------------------------------------*/
          numIter ++;
        }
        if( numIter%2 == 1 ) // Need to check parity for MPI overlaps to synchronize ( otherwise doesnt work )
        {
          helpFunc.swap( auxPtr );
          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
        }
        aux = *auxPtr;
        interfaceMap = *interfaceMapPtr;
#endif
      }


/**----------------------MPI-TO-DO---------------------------------------------**/
#ifdef HAVE_MPI
      if( TNL::MPI::GetSize() > 1 ) {
        getInfoFromNeighbours( calculatedBefore, calculateMPIAgain, distributedMesh );

        synchronizer.synchronize( aux );
      }
#endif
      // If we start the solver without MPI, we need calculated 0!
      if( TNL::MPI::GetSize() == 1 )
        calculatedBefore = 0;
    }
    iteration++;
  }
  Aux=auxPtr; // copy it for MakeSnapshot
}


// PROTECTED FUNCTIONS:

template< typename Real, typename Device, typename Index, typename Anisotropy >
bool
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
goThroughSweep( const StaticVector boundsFrom, const StaticVector boundsTo,
        MeshFunctionType& aux, const InterfaceMapType& interfaceMap,
        const AnisotropyPointer& anisotropy )
{
  bool calculated = false;
  const MeshType& mesh = aux.getMesh();
  const IndexType stepX = boundsFrom[0] < boundsTo[0]? 1 : -1;
  const IndexType stepY = boundsFrom[1] < boundsTo[1]? 1 : -1;

  typename MeshType::Cell cell( mesh );
  cell.refresh();

  for( cell.getCoordinates().y() = boundsFrom[1];
          TNL::abs( cell.getCoordinates().y() - boundsTo[1] ) > 0;
          cell.getCoordinates().y() += stepY )
  {
    for( cell.getCoordinates().x() = boundsFrom[0];
           TNL::abs( cell.getCoordinates().x() - boundsTo[0] ) > 0;
            cell.getCoordinates().x() += stepX )
    {
      cell.refresh();
      if( ! interfaceMap( cell ) )
      {
        calculated = this->updateCell( aux, cell ) || calculated;
      }
    }
  }
  return calculated;
}




#ifdef HAVE_MPI
template< typename Real, typename Device, typename Index, typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 2, Real, Device, Index >, Anisotropy >::
getInfoFromNeighbours( int& calculatedBefore, int& calculateMPIAgain,
                       const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh )
{
  int calculateFromNeighbours[4] = {0,0,0,0};
  const int *neighbours = distributedMesh.getNeighbors(); // Getting neighbors of distributed mesh
  MPI_Request *requestsInformation;
  requestsInformation = new MPI_Request[ distributedMesh.getNeighborsCount() ];

  int neighCount = 0; // should this thread calculate again?

  if( neighbours[0] != -1 ) // LEFT
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[0], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[0], 1, neighbours[0], 0, MPI_COMM_WORLD );
  }

  if( neighbours[1] != -1 ) // RIGHT
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[1], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[1], 1, neighbours[1], 0, MPI_COMM_WORLD );
  }

  if( neighbours[2] != -1 ) //UP
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[2], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[2], 1, neighbours[2], 0, MPI_COMM_WORLD  );
  }

  if( neighbours[5] != -1 ) //DOWN
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[5], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[3], 1, neighbours[5], 0, MPI_COMM_WORLD );
  }
  TNL::MPI::Waitall( requestsInformation, neighCount );

  TNL::MPI::Allreduce( &calculatedBefore, &calculatedBefore, 1, MPI_LOR,  MPI_COMM_WORLD );
  calculateMPIAgain = calculateFromNeighbours[0] || calculateFromNeighbours[1] ||
              calculateFromNeighbours[2] || calculateFromNeighbours[3];
}
#endif
