find_library(OMP_LIBRARY NAMES gomp HINTS $ENV{CONDA_PREFIX}/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib)
mark_as_advanced(OMP_LIBRARY)
add_library(openmp SHARED IMPORTED)
set_target_properties(openmp 
    PROPERTIES IMPORTED_LOCATION ${OMP_LIBRARY}
    INTERFACE_COMPILE_OPTIONS -fopenmp)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMP DEFAULT_MSG
    OMP_LIBRARY    
)