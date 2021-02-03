####################################################################################################
##################                     GHMC Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

find_package(pugixml CONFIG REQUIRED)

# Main executables
add_executable(load load.cc)

# Link libraries
target_link_libraries(load PRIVATE pugixml GHMC)
target_compile_options(load PRIVATE -Wall -Wextra -Wpedantic -O3)
