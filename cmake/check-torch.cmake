# Check for a possible PyTorch installation
find_package(Python QUIET)
if(Python_FOUND)
	if(EXISTS ${Python_SITELIB}/torch)
		list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/torch)
		# Fetch LibTorch _GLIBCXX_USE_CXX11_ABI value
		EXEC_PROGRAM(${Python_EXECUTABLE}
				ARGS "${PROJECT_SOURCE_DIR}/cmake/check-torch-cxx11-abi.py"
				OUTPUT_VARIABLE TORCH_USE_CXX11_ABI)
		message(STATUS "Found Torch _GLIBCXX_USE_CXX11_ABI = ${TORCH_USE_CXX11_ABI}")
	endif()
endif()

find_package(Torch REQUIRED)

# Default LibTorch _GLIBCXX_USE_CXX11_ABI value 0

if(NOT DEFINED TORCH_USE_CXX11_ABI)
	set(TORCH_USE_CXX11_ABI 0)
endif(NOT DEFINED TORCH_USE_CXX11_ABI)
message(STATUS "Torch _GLIBCXX_USE_CXX11_ABI = ${TORCH_USE_CXX11_ABI}")
