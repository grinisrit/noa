/*
 * Options.cu
 *
 *  Created on: Nov 24, 2014
 *      Author: yongchao
 */

#include "Options.h"

void Options::printUsage() {
	cerr << endl
			<< "LightSpMV (" << VERSION << ")"
			<< ": GPU-based sparse matrix-vector multiplication using CSR storate format"
			<< endl;
	cerr << "Usage: lightspmv -i matrix [options]" << endl << endl;
	cerr << "Options:" << endl;
	cerr << "Input:" << endl
			<< "\t-i <string> sparse matrix A file (in Matrix Market format)"
			<< endl
			<< "\t-x <string> vector X file (one element per line) [otherwise, set each element to 1.0]"
			<< endl
			<< "\t-y <string> vector Y file (one elemenet per line) [otherwise, set each element to 0.0]"
			<< endl << "Output:" << endl
			<< "\t-o <string> output file (one element per line) [otherwise, no output]"
			<< endl << "Compute:" << endl
			<< "\t-a <float> alpha value, default = " << _alpha << endl
			<< "\t-b <float> beta value, defualt = " << _beta << endl
			<< "\t-f <int> formula used, default = " << _formula << endl
			<< "\t    0: y = Ax" << endl << "\t    1: y = alpha * Ax + beta * y"
			<< endl << "\t-r <int> select the routine to use, default = "
			<< _routine << endl
			<< "\t    0: vector-based row dynamic distribution" << endl
			<< "\t    1: warp-based row dynamic distribution" << endl
			<< "\t-d <int> double-precision floating point, default = "
			<< (_singlePrecision ? 0 : 1) << endl
			<< "\t-g <int> index of the single GPU used, default = "
			<< _gpuIndex << endl
			<< "\t-m <int> number of SpMV iterations, default = " << _numIters
			<< endl << endl;
}
bool Options::parseArgs(int32_t argc, char* argv[]) {
	int32_t c;

	if (argc < 2) {
		printUsage();
		return false;
	}

	while ((c = getopt(argc, argv, "i:x:y:o:g:f:r:d:m:\n")) != -1) {
		switch (c) {
		case 'i':
			_mmFileName = optarg;
			break;
		case 'x':
			_vecXFileName = optarg;
			break;
		case 'y':
			_vecYFileName = optarg;
			break;
		case 'o':
			_outFileName = optarg;
			break;
		case 'a':
			_alpha = atof(optarg);
			break;
		case 'b':
			_beta = atof(optarg);
			break;
		case 'f':
			_formula = atoi(optarg);
			break;
		case 'g':
			_gpuIndex = atoi(optarg);
			if (_gpuIndex < 0) {
				_gpuIndex = 0;
			}
			break;
		case 'r':
			_routine = atoi(optarg);
			if (_routine < 0) {
				_routine = 0;
			}
			break;
		case 'd':
			_singlePrecision = atoi(optarg) ? false : true;
			break;
		case 'm':
			_numIters = atoi(optarg);
			if(_numIters < 1){
				_numIters = 1;
			}
			break;
		default:
			cerr << "Unknown parameter: " << optarg << endl;
			return false;
		}
	}

	/*check the file length*/
	if (_mmFileName.length() == 0) {
		cerr << "Matrix file should be specified" << endl;
		return false;
	}

	/*load the list of GPUs*/
	if (!getGPUs()) {
		return false;
	}

	/*load the matrix*/
	if (!loadMatrixMarketFile(_mmFileName.c_str())) {
		return false;
	}

	/*load vector X*/
	int64_t elementSize = _singlePrecision ? sizeof(float) : sizeof(double);
	int64_t numBytes = _numCols * elementSize;

	/*allocate space*/
	cudaMallocHost(&_vectorX, numBytes);
	CudaCheckError();

	/*load the vector X*/
	if (_vecXFileName.length() == 0) {
		/*initialize X*/
		cerr << "Initialize each element of vector X to 1.0" << endl;
		if (_singlePrecision) {
			float* p = (float*) _vectorX;
			for (uint32_t i = 0; i < _numCols; ++i) {
				p[i] = 1.0;
			}
		} else {
			double* p = (double*) _vectorX;
			for (uint32_t i = 0; i < _numCols; ++i) {
				p[i] = 1.0;
			}
		}
	} else {
		cerr << "Load vector X from file" << endl;
		/*could not get the data*/
		if (!loadVector(_vecXFileName, _vectorX, _numCols)) {
			return false;
		}
	}

	/*load vector Y*/
	numBytes = _numRows * elementSize;

	/*allocate space*/
	cudaMallocHost(&_vectorY, numBytes);
	CudaCheckError();

	/*load the vector Y*/
	if (_vecYFileName.length() == 0) {
		/*initialize Y*/
		cerr << "Initialize each element of vector Y to 0" << endl;

		memset(_vectorY, 0, numBytes);
	} else {
		cerr << "Load vector Y from file" << endl;
		/*could not get the data*/
		if (!loadVector(_vecYFileName, _vectorY, _numRows)) {
			return false;
		}
	}

	return true;
}
/*convert the matrix market format to CSR*/
bool Options::loadMatrixMarketFile(const char* fileName) {
	uint64_t numBytes;

	cerr << "loading sparse matrix" << endl;
	if (_singlePrecision) {
		/*create an empty CSR sparse matrix object*/
		cusp::csr_matrix<uint32_t, float, cusp::host_memory> matrix;

		// load a matrix stored in MatrixMarket format
		cusp::io::read_matrix_market_file(matrix, fileName);

		/*save the matrix information*/
		_numRows = matrix.num_rows;
		_numCols = matrix.num_cols;
		_numValues = matrix.num_entries;

		/*reserve memory*/
		cudaMallocHost(&_rowOffsets, (_numRows + 1) * sizeof(uint32_t));
		CudaCheckError();

		cudaMallocHost(&_colIndexValues, _numValues * sizeof(uint32_t));
		CudaCheckError();

		cudaMallocHost(&_numericalValues, _numValues * sizeof(float));
		CudaCheckError();

		/*copy the elements*/
		numBytes = (_numRows + 1) * sizeof(uint32_t);
		cudaMemcpy(_rowOffsets, &matrix.row_offsets[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();

		numBytes = _numValues * sizeof(uint32_t);
		cudaMemcpy(_colIndexValues, &matrix.column_indices[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();

		numBytes = _numValues * sizeof(float);
		cudaMemcpy(_numericalValues, &matrix.values[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();
	} else {
		/*create an empty CSR sparse matrix object*/
		cusp::csr_matrix<uint32_t, double, cusp::host_memory> matrix;

		// load a matrix stored in MatrixMarket format
		cusp::io::read_matrix_market_file(matrix, fileName);

		/*save the matrix information*/
		_numRows = matrix.num_rows;
		_numCols = matrix.num_cols;
		_numValues = matrix.num_entries;

		/*reserve memory*/
		cudaMallocHost(&_rowOffsets, (_numRows + 1) * sizeof(uint32_t));
		CudaCheckError();

		cudaMallocHost(&_colIndexValues, _numValues * sizeof(uint32_t));
		CudaCheckError();

		cudaMallocHost(&_numericalValues, _numValues * sizeof(double));
		CudaCheckError();

		/*copy the elements*/
		numBytes = (_numRows + 1) * sizeof(uint32_t);
		cudaMemcpy(_rowOffsets, &matrix.row_offsets[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();

		numBytes = _numValues * sizeof(uint32_t);
		cudaMemcpy(_colIndexValues, &matrix.column_indices[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();

		numBytes = _numValues * sizeof(double);
		cudaMemcpy(_numericalValues, &matrix.values[0], numBytes,
				cudaMemcpyHostToHost);
		CudaCheckError();
	}

	return true;
}
bool Options::loadVector(const string& fileName, void* vector,
		const uint32_t maxNumValues) {
	char buffer[1024];
	FILE* file;
	uint32_t pos;
	float* fptr = (float*) vector;
	double* dptr = (double*) vector;

	cerr << "loading vector X" << endl;
	/*open the file*/
	if (fileName.length() == 0) {
		return false;
	}
	file = fopen(fileName.c_str(), "r");
	if (!file) {
		cerr << "Failed to open file " << fileName << endl;
		return false;
	}

	/*read the file*/
	pos = 0;
	while (fgets(buffer, 1023, file)) {
		/*remove the end of line*/
		for (int32_t i = strlen(buffer) - 1;
				i >= 0 && (buffer[i] == '\n' || buffer[i] == '\r'); --i) {
			buffer[i] = '\0';
		}
		if (strlen(buffer) == 0) {
			continue;
		}

		/*get the number and save to vector*/
		if (pos >= maxNumValues) {
			/*already have enough numbers*/
			break;
		}
		if (_singlePrecision) {
			float value;
			sscanf(buffer, "%f", &value);
			fptr[pos++] = value;
		} else {
			double value;
			sscanf(buffer, "%lf", &value);
			dptr[pos++] = value;
		}
	}
	if (pos < maxNumValues) {
		cerr << "Do not have enough numbers in the file" << endl;
		return false;
	}
	cerr << "Finished loading vector X" << endl;
	return true;
}
void Options::getRowSizeVariance() {
	double rowStart;
	uint32_t rowEnd;

	/*compute the variance*/
	_variance = 0;
	_mean = rint((double) _numValues / _numRows);
	rowStart = _rowOffsets[0];
	for (uint32_t i = 1; i <= _numRows; ++i) {
		rowEnd = _rowOffsets[i];
		_variance += (rowEnd - rowStart - _mean) * (rowEnd - rowStart - _mean);
		rowStart = rowEnd;
	}
	_variance = rint(sqrt(_variance / (_numRows > 1 ? _numRows - 1 : 1)));

	/*information*/
	cerr << "Rows: " << _numRows << " Cols: " << _numCols << " Elements: "
			<< _numValues << " Mean: " << _mean << " Standard deviation: "
			<< _variance << endl;
}
bool Options::getGPUs() {
	int32_t numGPUs;

	/*get the number of GPUs*/
	if (cudaGetDeviceCount(&numGPUs) != cudaSuccess) {
		cerr << "No CUDA-enabled GPU is available in the host" << endl;
		return false;
	}

#if defined(HAVE_SM_35)
	cerr << "Require GPUs with compute capability >= 3.5" << endl;
#else
	cerr << "Require GPUs with compute capability >= 3.0" << endl;
#endif

	/*iterate each GPU*/
	cudaDeviceProp prop;
	for (int32_t i = 0; i < numGPUs; ++i) {

		/*get the property of the device*/
		cudaGetDeviceProperties(&prop, i);

		/*check the major of the GPU*/
#if defined(HAVE_SM_35)
		if ((prop.major * 10 + prop.minor) >= 35) {
#else
		if ((prop.major * 10 + prop.minor) >= 30) {
#endif
			cerr << "GPU " << _gpus.size() << ": " << prop.name
					<< " (capability " << prop.major << "." << prop.minor << ")"
					<< endl;

			/*save the Kepler GPU*/
			_gpus.push_back(make_pair(i, prop));
		}
	}
	/*check the number of qualified GPUs*/
	if (_gpus.size() == 0) {
		cerr << "No qualified GPU is available" << endl;
		return false;
	}

	/*check the GPU index*/

	/*reset the number of GPUs*/
	if (_gpuIndex >= (int32_t) _gpus.size()) {
		_gpuIndex = _gpus.size() - 1;
	}
	if (_gpuIndex < 0) {
		_gpuIndex = 0;
	}

	/*move the selected gpu to the first*/
	swap(_gpus[0], _gpus[_gpuIndex]);

	return true;
}
