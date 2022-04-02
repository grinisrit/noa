# Check for a possible PyTorch installation
find_package(Python QUIET)
if(Python_FOUND)
	if(EXISTS ${Python_SITELIB}/torch)
		list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/torch)
	endif()
endif()

find_package(Torch REQUIRED)

# Get LibTorch _GLIBCXX_USE_CXX11_ABI value
EXEC_PROGRAM(python
	ARGS "${PROJECT_SOURCE_DIR}/cmake/check-torch-cxx11-abi.py"
	OUTPUT_VARIABLE TORCH_USE_CXX11_ABI)
message(STATUS "Torch _GLIBCXX_USE_CXX11_ABI = ${TORCH_USE_CXX11_ABI}")
