# Check for a possible PyTorch installation
find_package(Python QUIET)
if(Python_FOUND)
	if(EXISTS ${Python_SITELIB}/torch)
		list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/torch)
	endif()
endif()

find_package(Torch REQUIRED)