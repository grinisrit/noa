#pragma once

#include <TNL/Functions/MeshFunction.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/MPI/Wrappers.h>

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
FastSweepingMethod()
: maxIterations( 1 )
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
const Index&
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
getMaxIterations() const
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
setMaxIterations( const IndexType& maxIterations )
{

}

template< typename Real,
        typename Device,
        typename Index,
        typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
solve( const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh,
       const MeshPointer& mesh,
       MeshFunctionPointer& Aux,
       const AnisotropyPointer& anisotropy,
       MeshFunctionPointer& u )
{
  MeshFunctionPointer auxPtr;
  InterfaceMapPointer interfaceMapPtr;
  auxPtr->setMesh( mesh );
  interfaceMapPtr->setMesh( mesh );

  // getting overlaps ( WITHOUT MPI SHOULD BE 0 )
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
  auxPtr->write( "aux", "aux-ini.vti" );

  typename MeshType::Cell cell( *mesh );

  IndexType iteration( 0 );
  MeshFunctionType aux = *auxPtr;
  InterfaceMapType interfaceMap = * interfaceMapPtr;
  synchronizer.setDistributedGrid( &distributedMesh );
  synchronizer.synchronize( aux ); //synchronization of intial conditions

  while( iteration < this->maxIterations )
  {
    // indicates weather we calculated in the last passage of the while cycle
    // calculatedBefore is same for all ranks
    // without MPI should be FALSE at the end of while cycle body
    int calculatedBefore = 1;

    // indicates if the MPI process should calculate again in upcoming passage of cycle
    // calculateMPIAgain is a value that can differ in every rank
    // without MPI should be FALSE at the end of while cycle body
    int calculateMPIAgain = 1;

    while( calculatedBefore )
    {
      calculatedBefore = 0;

      if( std::is_same< DeviceType, Devices::Host >::value && calculateMPIAgain ) // should we calculate in Host?
      {
        calculateMPIAgain = 0;

/** HERE IS FSM FOR OPENMP (NO MPI) - isnt worthy */
        /*int numThreadsPerBlock = -1;

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
         int numBlocksZ = mesh->getDimensions().z() / numThreadsPerBlock + (mesh->getDimensions().z() % numThreadsPerBlock != 0 ? 1:0);
         //std::cout << "numBlocksX = " << numBlocksX << std::endl;

         //Real **sArray = new Real*[numBlocksX*numBlocksY];
         // for( int i = 0; i < numBlocksX * numBlocksY; i++ )
         // sArray[ i ] = new Real [ (numThreadsPerBlock + 2)*(numThreadsPerBlock + 2)];

         ArrayContainer BlockIterHost;
         BlockIterHost.setSize( numBlocksX * numBlocksY * numBlocksZ );
         BlockIterHost.setValue( 1 );
         int IsCalculationDone = 1;

         MeshFunctionPointer helpFunc( mesh );
         MeshFunctionPointer helpFunc1( mesh );
         helpFunc1 = auxPtr;
         auxPtr = helpFunc;
         helpFunc = helpFunc1;
         //std::cout<< "Size = " << BlockIterHost.getSize() << std::endl;
         //for( int k = numBlocksX-1; k >-1; k-- ){
         //for( int l = 0; l < numBlocksY; l++ ){
         // std::cout<< BlockIterHost[ l*numBlocksX  + k ];
         // }
         // std::cout<<std::endl;
         // }
         // std::cout<<std::endl;
         unsigned int numWhile = 0;
         while( IsCalculationDone  )
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


         this->getNeighbours( BlockIterHost, numBlocksX, numBlocksY, numBlocksZ );

         //string s( "aux-"+ std::to_string(numWhile) + ".vti");
         //aux.write( "aux", s );
         }
         if( numWhile == 1 ){
         auxPtr = helpFunc;
         }
         aux = *auxPtr;*/
/**------------------------------------------------------------------------------*/


/** HERE IS FSM WITH MPI AND WITHOUT MPI */
        StaticVector boundsFrom; StaticVector boundsTo;

    // TOP, NORTH and EAST
        boundsFrom[2] = vecLowerOverlaps[2]; boundsTo[2] = mesh->getDimensions().z() - vecUpperOverlaps[2];
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        calculatedBefore = goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // TOP, NORTH and WEST
        boundsFrom[2] = vecLowerOverlaps[2]; boundsTo[2] = mesh->getDimensions().z() - vecUpperOverlaps[2];
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // TOP, SOUTH and EAST
        boundsFrom[2] = vecLowerOverlaps[2]; boundsTo[2] = mesh->getDimensions().z() - vecUpperOverlaps[2];
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // TOP, SOUTH and WEST
        boundsFrom[2] = vecLowerOverlaps[2]; boundsTo[2] = mesh->getDimensions().z() - vecUpperOverlaps[2];
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // BOTTOM, NOTH and EAST
        boundsFrom[2] = mesh->getDimensions().z() - 1 - vecUpperOverlaps[2]; boundsTo[2] = - 1 + vecLowerOverlaps[2];
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // BOTTOM, NOTH and WEST
        boundsFrom[2] = mesh->getDimensions().z() - 1 - vecUpperOverlaps[2]; boundsTo[2] = - 1 + vecLowerOverlaps[2];
        boundsFrom[1] = vecLowerOverlaps[1]; boundsTo[1] = mesh->getDimensions().y() - vecUpperOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // BOTTOM, SOUTH and EAST
        boundsFrom[2] = mesh->getDimensions().z() - 1 - vecUpperOverlaps[2]; boundsTo[2] = - 1 + vecLowerOverlaps[2];
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = vecLowerOverlaps[0]; boundsTo[0] = mesh->getDimensions().x() - vecUpperOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );

    // BOTTOM, SOUTH and WEST
        boundsFrom[2] = mesh->getDimensions().z() - 1 - vecUpperOverlaps[2]; boundsTo[2] = - 1 + vecLowerOverlaps[2];
        boundsFrom[1] = mesh->getDimensions().y() - 1 - vecUpperOverlaps[1]; boundsTo[1] = - 1 + vecLowerOverlaps[1];
        boundsFrom[0] = mesh->getDimensions().x() - 1 - vecUpperOverlaps[0]; boundsTo[0] = - 1 + vecLowerOverlaps[0];
        goThroughSweep( boundsFrom, boundsTo, aux, interfaceMap, anisotropy );


  /**----------------------------------------------------------------------------------*/
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value && calculateMPIAgain )
      {
#ifdef HAVE_CUDA
        // cudaBlockSize is a size of blocks. It's the number raised to the 3 power.
        // the number should be less than 10^3 (num of threads in one grid is maximally 1024)
        // IF YOU CHANGE THIS, YOU NEED TO CHANGE THE TEMPLATE PARAMETER IN CudaUpdateCellCaller (The Number + 2)
        const int cudaBlockSize( 8 );

        // Getting the number of blocks in grid in each direction (without overlaps bcs we dont calculate on overlaps)
        int numBlocksX = Cuda::getNumberOfBlocks( mesh->getDimensions().x() - vecLowerOverlaps[0] - vecUpperOverlaps[0], cudaBlockSize );
        int numBlocksY = Cuda::getNumberOfBlocks( mesh->getDimensions().y() - vecLowerOverlaps[1] - vecUpperOverlaps[1], cudaBlockSize );
        int numBlocksZ = Cuda::getNumberOfBlocks( mesh->getDimensions().z() - vecLowerOverlaps[2] - vecUpperOverlaps[2], cudaBlockSize );
        if( cudaBlockSize * cudaBlockSize * cudaBlockSize > 1024 || numBlocksX > 1024 || numBlocksY > 1024 || numBlocksZ > 64 )
          std::cout << "Invalid kernel call. Dimensions of grid are max: [1024,1024,64], and maximum threads per block are 1024!" << std::endl;

        // Making the variables for global function CudaUpdateCellCaller.
        dim3 blockSize( cudaBlockSize, cudaBlockSize, cudaBlockSize );
        dim3 gridSize( numBlocksX, numBlocksY, numBlocksZ );

        BaseType ptr; // tnlDirectEikonalMethodBase type for calling of function inside CudaUpdateCellCaller


        int BlockIterD = 1; //variable that tells us weather we should calculate the main cuda body again

        // Array containing information about each block in grid, answering question (Have we calculated in this block?)
        TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterDevice( numBlocksX * numBlocksY * numBlocksZ );
        BlockIterDevice.setValue( 1 ); // calculate all in the first passage

        // Helping Array for GetNeighbours3D
        TNL::Containers::Array< int, Devices::Cuda, IndexType > BlockIterPom( numBlocksX * numBlocksY * numBlocksZ );
        BlockIterPom.setValue( 0 ); //doesnt matter what number



        // number of neighbours in one block (1024 threads) for GetNeighbours3D
        int nBlocksNeigh = ( numBlocksX * numBlocksY * numBlocksZ )/1024 + ((( numBlocksX * numBlocksY * numBlocksZ )%1024 != 0) ? 1:0);


        //MeshFunctionPointer helpFunc1( mesh );
        Containers::Vector< RealType, DeviceType, IndexType > helpVec;
        helpVec.setLike( auxPtr.template getData().getData() );
        MeshFunctionPointer helpFunc;
        helpFunc->bind( mesh, helpVec );
        helpFunc.template modifyData() = auxPtr.template getData();
        Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();

        int numIter = 0; // number of passages of following while cycle

        while( BlockIterD ) //main body of cuda code
        {

          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
          // main function that calculates all values in each blocks
          // calculated values are in helpFunc
          CudaUpdateCellCaller< 10 ><<< gridSize, blockSize >>>( ptr,
                  interfaceMapPtr.template getData< Device >(),
                  auxPtr.template getData< Device>(),
                  helpFunc.template modifyData< Device>(),
                  BlockIterDevice.getView(), vecLowerOverlaps, vecUpperOverlaps );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          // Switching pointers to helpFunc and auxPtr so real results are in memory of helpFunc but here under variable auxPtr
          auxPtr.swap( helpFunc );

          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
          // Neighbours of blocks that calculatedBefore in this passage should calculate in the next!
          // BlockIterDevice contains blocks that calculatedBefore in this passage and BlockIterPom those that should calculate in next (are neighbours)
          GetNeighbours<<< nBlocksNeigh, 1024 >>>( BlockIterDevice.getView(), BlockIterPom.getView(), numBlocksX, numBlocksY, numBlocksZ );
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;
          BlockIterDevice = BlockIterPom;
          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();

          // contains(...) is actually parallel reduction implemented in TNL
          BlockIterD = Algorithms::contains( BlockIterDevice, 1);
          cudaDeviceSynchronize();
          TNL_CHECK_CUDA_DEVICE;

          numIter++;
          if( BlockIterD ){
            // if we calculated in this passage, we should send the info via MPI so neighbours should calculate after synchronization
            calculatedBefore = 1;
          }
        }
        if( numIter%2 == 1 ){

          // We need auxPtr to point on memory of original auxPtr (not to helpFunc)
          // last passage of previous while cycle didnt calculate any number anyway so switching names doesnt effect values
          auxPtr.swap( helpFunc );
          Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
        }
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        aux = *auxPtr;
        interfaceMap = *interfaceMapPtr;
#endif
      }

#ifdef HAVE_MPI
      if( TNL::MPI::GetSize() > 1 )
      {
        getInfoFromNeighbours( calculatedBefore, calculateMPIAgain, distributedMesh );

        // synchronizate the overlaps
        synchronizer.synchronize( aux );

      }
#endif

      // If we start the solver without MPI, we need calculatedBefore 0!
      // otherwise we would go throw the FSM code and CUDA FSM code again uselessly
      if( TNL::MPI::GetSize() == 1 )
        calculatedBefore = 0;
    }
    //aux.write( "aux", "aux-8.vti" );
    iteration++;

  }
  // Saving the results into Aux for MakeSnapshot function.
  Aux = auxPtr;
  aux.write( "aux", "aux-final.vti" );
}

// PROTECTED FUNCTIONS:

template< typename Real, typename Device, typename Index, typename Anisotropy >
bool
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
goThroughSweep( const StaticVector boundsFrom, const StaticVector boundsTo,
        MeshFunctionType& aux, const InterfaceMapType& interfaceMap,
        const AnisotropyPointer& anisotropy )
{
  bool calculated = false;
  const MeshType& mesh = aux.getMesh();
  const IndexType stepX = boundsFrom[0] < boundsTo[0]? 1 : -1;
  const IndexType stepY = boundsFrom[1] < boundsTo[1]? 1 : -1;
  const IndexType stepZ = boundsFrom[2] < boundsTo[2]? 1 : -1;

  typename MeshType::Cell cell( mesh );
  cell.refresh();

  for( cell.getCoordinates().z() = boundsFrom[2];
          TNL::abs( cell.getCoordinates().z() - boundsTo[2] ) > 0;
          cell.getCoordinates().z() += stepZ )
  {
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
  }
  return calculated;
}




#ifdef HAVE_MPI
template< typename Real, typename Device, typename Index, typename Anisotropy >
void
FastSweepingMethod< Meshes::Grid< 3, Real, Device, Index >, Anisotropy >::
getInfoFromNeighbours( int& calculatedBefore, int& calculateMPIAgain,
                       const Meshes::DistributedMeshes::DistributedMesh< MeshType >& distributedMesh )
{
  int calculateFromNeighbours[6] = {0,0,0,0,0,0};

  const int *neighbours = distributedMesh.getNeighbors(); // Getting neighbors of distributed mesh
  MPI_Request *requestsInformation;
  requestsInformation = new MPI_Request[ distributedMesh.getNeighborsCount() ];

  int neighCount = 0; // should this thread calculate again?

  if( neighbours[0] != -1 ) // WEST
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[0], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[0], 1, neighbours[0], 0, MPI_COMM_WORLD );
  }

  if( neighbours[1] != -1 ) // EAST
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[1], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[1], 1, neighbours[1], 0, MPI_COMM_WORLD );
  }

  if( neighbours[2] != -1 ) //NORTH
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[2], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[2], 1, neighbours[2], 0, MPI_COMM_WORLD );
  }

  if( neighbours[5] != -1 ) //SOUTH
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[5], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[3], 1, neighbours[5], 0, MPI_COMM_WORLD );
  }

  if( neighbours[8] != -1 ) // TOP
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[8], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[4], 1, neighbours[8], 0, MPI_COMM_WORLD );
  }

  if( neighbours[17] != -1 ) //BOTTOM
  {
    requestsInformation[neighCount++] =
            TNL::MPI::Isend( &calculatedBefore, 1, neighbours[17], 0, MPI_COMM_WORLD );
    requestsInformation[neighCount++] =
            TNL::MPI::Irecv( &calculateFromNeighbours[5], 1, neighbours[17], 0, MPI_COMM_WORLD );
  }

  TNL::MPI::Waitall( requestsInformation, neighCount );

  TNL::MPI::Allreduce( &calculatedBefore, &calculatedBefore, 1, MPI_LOR,  MPI_COMM_WORLD );
  calculateMPIAgain = calculateFromNeighbours[0] || calculateFromNeighbours[1] ||
                      calculateFromNeighbours[2] || calculateFromNeighbours[3] ||
                      calculateFromNeighbours[4] || calculateFromNeighbours[5];
}
#endif
