#include "quad-test-conf.h"
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Matrices/CSR.h>
//#include "../../src/matrix/CSR.h"
#include "Quadcpp.h"

int main(int argc, char* argv[]) {
	Config::ParameterContainer parameters;
	Config::ConfigDescription conf_desc;
	if(conf_desc.ParseConfigDescription(CONFIG_FILE) != 0)
		return EXIT_FAILURE;
	if(!parseCommandLine(argc, argv, conf_desc, parameters)) {
		conf_desc.PrintUsage(argv[ 0 ]);
		return EXIT_FAILURE;
	}

	String inputFile = parameters.getParameter <String> ("input-file");
	File binaryFile;
	if(! binaryFile.open(inputFile, File::Mode::In)) {
		cerr << "I am not able to open the file " << inputFile << "." << std::endl;
		return 1;
	}
	CSR <double> doubleMatrix("double");
	if(! doubleMatrix.load(binaryFile)) {
		cerr << "Unable to restore the CSR matrix." << std::endl;
		return 1;
	}
	binaryFile.close();
	
	CSR <QuadDouble> quadMatrix("quad");
	quadMatrix = doubleMatrix;
	return EXIT_SUCCESS;
}
