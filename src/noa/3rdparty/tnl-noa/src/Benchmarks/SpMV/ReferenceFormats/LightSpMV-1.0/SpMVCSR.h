/*
 * SpMVCSR.h
 *
 *  Created on: Nov 25, 2014
 *      Author: yongchao
 */

#ifndef SPMVCSR_H_
#define SPMVCSR_H_
#include "Types.h"

#pragma once

extern __constant__ uint32_t _cudaNumRows;
extern __constant__ uint32_t _cudaNumCols;

namespace spmv_csr {

/*device functions*/
template < typename T>
__device__ inline T shfl_down_64bits(T var, int32_t srcLane,
		int32_t width) {

	int2 a = *reinterpret_cast<int2*>(&var);

	/*exchange the data*/
	a.x = __shfl_down_sync(0xffffffff,a.x, srcLane, width);
	a.y = __shfl_down_sync(0xffffffff,a.y, srcLane, width);
	
	return *reinterpret_cast<T*>(&a);
}

/*macro to get the X value*/
__device__ inline float FLOAT_VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index){
	return tex1Dfetch<float>(vectorX, index);
}
__device__ inline float FLOAT_VECTOR_GET (const float* __restrict vectorX, uint32_t index){
	return vectorX[index];
}

__device__ inline double DOUBLE_VECTOR_GET (const cudaTextureObject_t vectorX, uint32_t index){
	/*load the data*/
	int2 v = tex1Dfetch<int2>(vectorX, index);

	/*convert to double*/
	return __hiloint2double(v.y, v.x);
}
__device__ inline double DOUBLE_VECTOR_GET (const double* __restrict vectorX, uint32_t index){
	return vectorX[index];
}


/*32-bit*/
template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef FLOAT_USE_TEXTURE_MEMORY
__global__ void csr32DynamicWarp(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX,  T* vectorY) {
#else
	__global__ void csr32DynamicWarp(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
			const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY) {
#endif
	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;	/*vector index in the warp*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down_sync(0xffffffff,sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if(warpLaneId == 0){
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	}/*while*/
}

/*vector-based row dynamic distribution*/
template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef FLOAT_USE_TEXTURE_MEMORY
__global__ void csr32DynamicVector(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX, T* vectorY) {
#else
	__global__ void csr32DynamicVector(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
			const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY) {
#endif

	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the block*/
	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (laneId == 0) {
		row = atomicAdd(cudaRowCounter, 1);
	}
	/*broadcast the value to other lanes from lane 0*/
	row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += __shfl_down_sync(0xffffffff,sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;

			/*get a new row index*/
			row = atomicAdd(cudaRowCounter, 1);
		}
		row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);
	}/*while*/
}

	/*32-bit*/
	template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
	#ifdef FLOAT_USE_TEXTURE_MEMORY
	__global__ void csr32DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
			const T* __restrict numericalValues, const cudaTextureObject_t vectorX,  T* vectorY, const T alpha, const T beta) {
	#else
		__global__ void csr32DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
				const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY, const T alpha, const T beta) {
	#endif
		uint32_t i;
		T sum;
		uint32_t row = 0;
		uint32_t rowStart, rowEnd;
		const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
		const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
		const uint32_t warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
		const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;	/*vector index in the warp*/

		__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

		/*get the row index*/
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

		/*check the row range*/
		while (row < _cudaNumRows) {

			/*use two threads to fetch the row offset*/
			if (laneId < 2) {
				space[vectorId][laneId] = rowOffsets[row + laneId];
			}
			rowStart = space[vectorId][0];
			rowEnd = space[vectorId][1];

			/*there are non-zero elements in the current row*/
			sum = 0;
			/*compute dot product*/
			if (THREADS_PER_VECTOR == 32) {

				/*ensure aligned memory access*/
				i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

				/*process the unaligned part*/
				if (i >= rowStart && i < rowEnd) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}

					/*process the aligned part*/
				for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			} else {
				/*regardless of the global memory access alignment*/
				for (i = rowStart + laneId; i < rowEnd; i +=
						THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			/*intra-vector reduction*/
			sum *= alpha;
			for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += __shfl_down_sync(0xffffffff,sum, i, THREADS_PER_VECTOR);
			}

			/*save the results and get a new row*/
			if (laneId == 0) {
				/*save the results*/
				vectorY[row] = sum + beta * vectorY[row];
			}

			/*get a new row index*/
			if(warpLaneId == 0){
				row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
			}
			/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
			row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

		}/*while*/
	}

	/*vector-based row dynamic distribution*/
	template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
	#ifdef FLOAT_USE_TEXTURE_MEMORY
	__global__ void csr32DynamicVectorBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
			const T* __restrict numericalValues, const cudaTextureObject_t vectorX, T* vectorY, const T alpha, const T beta) {
	#else
		__global__ void csr32DynamicVectorBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
				const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY, const T alpha, const T beta) {
	#endif

		uint32_t i;
		T sum;
		uint32_t row = 0;
		uint32_t rowStart, rowEnd;
		const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
		const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the block*/
		__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

		/*get the row index*/
		if (laneId == 0) {
			row = atomicAdd(cudaRowCounter, 1);
		}
		/*broadcast the value to other lanes from lane 0*/
		row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);

		/*check the row range*/
		while (row < _cudaNumRows) {

			/*use two threads to fetch the row offset*/
			if (laneId < 2) {
				space[vectorId][laneId] = rowOffsets[row + laneId];
			}
			rowStart = space[vectorId][0];
			rowEnd = space[vectorId][1];

			/*there are non-zero elements in the current row*/
			sum = 0;
			/*compute dot product*/
			if (THREADS_PER_VECTOR == 32) {

				/*ensure aligned memory access*/
				i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

				/*process the unaligned part*/
				if (i >= rowStart && i < rowEnd) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}

					/*process the aligned part*/
				for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			} else {
				/*regardless of the global memory access alignment*/
				for (i = rowStart + laneId; i < rowEnd; i +=
						THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			/*intra-vector reduction*/
			sum *= alpha;
			for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += __shfl_down_sync(0xffffffff,sum, i, THREADS_PER_VECTOR);
			}

			/*save the results and get a new row*/
			if (laneId == 0) {
				/*save the results*/
				vectorY[row] = sum + beta * vectorY[row];

				/*get a new row index*/
				row = atomicAdd(cudaRowCounter, 1);
			}
			row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);
		}/*while*/
	}

/*64-bit functions*/
template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef DOUBLE_USE_TEXTURE_MEMORY
__global__ void csr64DynamicVector(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX, T* vectorY)
#else
__global__ void csr64DynamicVector(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY)
#endif
{
	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the block*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (laneId == 0) {
		row = atomicAdd(cudaRowCounter, 1);
	}
	/*broadcast the value to other lanes from lane 0*/
	row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;

			/*get a new row index*/
			row = atomicAdd(cudaRowCounter, 1);
		}
		row = __shfl_sync( 0xffffffff, row, 0, THREADS_PER_VECTOR);
	}/*while*/
}

template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef DOUBLE_USE_TEXTURE_MEMORY
__global__ void csr64DynamicWarp(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX, T* vectorY)
#else
__global__ void csr64DynamicWarp(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX, T* vectorY)
#endif
{
	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;	/*vector index in the warp*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if(warpLaneId == 0){
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	}/*while*/
}

/*64-bit functions*/
template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef DOUBLE_USE_TEXTURE_MEMORY
__global__ void csr64DynamicVectorBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX, const T* __restrict inVectorY, T* vectorY, const T alpha, const T beta)
#else
__global__ void csr64DynamicVectorBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX, const T* __restrict inVectorY, T* vectorY, const T alpha, const T beta)
#endif
{
	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the block*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (laneId == 0) {
		row = atomicAdd(cudaRowCounter, 1);
	}
	/*broadcast the value to other lanes from lane 0*/
	row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * DOUBLE_VECTOR_GET(inVectorY, row);

			/*get a new row index*/
			row = atomicAdd(cudaRowCounter, 1);
		}
		row = __shfl_sync(0xffffffff,row, 0, THREADS_PER_VECTOR);
	}/*while*/
}

template < typename T, uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
#ifdef DOUBLE_USE_TEXTURE_MEMORY
__global__ void csr64DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const cudaTextureObject_t vectorX, const T* __restrict inVectorY, T* vectorY, const T alpha, const T beta)
#else
__global__ void csr64DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter, const uint32_t* __restrict rowOffsets, const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const T* __restrict vectorX, const T* __restrict inVectorY, T* vectorY, const T alpha, const T beta)
#endif
{
	uint32_t i;
	T sum;
	uint32_t row = 0;
	uint32_t rowStart, rowEnd;
	const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const uint32_t warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
	const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;	/*vector index in the warp*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp*/
	row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}

				/*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i +=
					THREADS_PER_VECTOR) {
				sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
			}
		}

		/*intra-vector reduction*/
		sum *= alpha;
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum + beta * DOUBLE_VECTOR_GET(inVectorY, row);
		}

		/*get a new row index*/
		if(warpLaneId == 0){
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the value to other threads in the same warp*/
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	}/*while*/
}


}/*namespace*/

#endif /* SPMVCSR_H_ */
