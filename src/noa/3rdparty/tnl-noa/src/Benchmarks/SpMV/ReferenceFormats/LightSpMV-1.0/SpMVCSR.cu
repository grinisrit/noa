/*
 * SpMVCSR.cu
 *
 *  Created on: Nov 25, 2014
 *      Author: yongchao
 */
#include "SpMVCSR.h"

/*device variables*/
__constant__ uint32_t _cudaNumRows;
__constant__ uint32_t _cudaNumCols;

