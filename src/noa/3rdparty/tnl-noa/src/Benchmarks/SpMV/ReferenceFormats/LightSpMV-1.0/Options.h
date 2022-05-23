/*
 * Options.h
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include "Types.h"
//#include <cusp/io/matrix_market.h>

struct Options {
	Options() {

		/*input*/
		_routine = 1;
		_formula = 1;
		_numIters = 1000;
		_singlePrecision = true;

		/*matrix data*/
		_numRows = 0;
		_numCols = 0;
		_rowOffsets = NULL;
		_numValues = 0;
		_colIndexValues = NULL;
		_numericalValues = NULL;
		_alpha = 1.0;
		_beta = 1.0;

		/*vector data*/
		_vectorX = NULL;
		_vectorY = NULL;

		/*the number of GPUs*/
		_numGPUs = 1;

		/*GPU index used*/
		_gpuIndex = 0;

		/*for debug*/
		_mean = 0;
		_variance = 0;
	}
	~Options() {
		if (_rowOffsets) {
			cudaFreeHost(_rowOffsets);
		}
		if (_colIndexValues) {
			cudaFreeHost(_colIndexValues);
		}
		if (_numericalValues) {
			cudaFreeHost(_numericalValues);
		}

		if (_vectorX) {
			cudaFreeHost(_vectorX);
		}
		if (_vectorY) {
			cudaFreeHost(_vectorY);
		}
	}

	/*parse parameters*/
	bool parseArgs(int32_t argc, char* argv[]);

	/*load Matrix Market file*/
	bool loadMatrixMarketFile(const char* fileName);

	/*load vector*/
	bool loadVector(const string& fileName, void* vector,
			const uint32_t maxNumValues);

	/*print out usage*/
	void printUsage();

	/*get row distribution*/
	void getRowSizeVariance();

	/*retrieve GPU list*/
	bool getGPUs();

	/*input files*/
	string _mmFileName;
	string _vecXFileName;
	string _vecYFileName;
	string _outFileName;
	bool _singlePrecision;
	int32_t _routine;
	int32_t _formula;
	int32_t _numIters;
	double _alpha;
	double _beta;

	/*for debugging*/
	double _mean;
	double _variance;

	/*matrix data*/
	uint32_t _numRows;
	uint32_t _numCols;
	uint32_t *_rowOffsets;
	uint32_t _numValues;
	uint32_t *_colIndexValues;
	void *_numericalValues;

	/*vector data*/
	void *_vectorX;
	void *_vectorY;

	/*number of GPUs to be used*/
	int32_t _numGPUs;

	/*GPU index used*/
	int32_t _gpuIndex;

	/*GPU device list*/
	vector<pair<int32_t, struct cudaDeviceProp> > _gpus;
};

#endif /* OPTIONS_H_ */
