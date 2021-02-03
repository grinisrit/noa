####################################################################################################
################                     GHMC Apps: sampler examples                    ################
####################################################################################################

message(STATUS "Building sampler apps required")

# Main executables
add_executable(normal-cpu normal-cpu.cc)

# Link libraries
target_link_libraries(normal-cpu PRIVATE GHMC)
target_compile_options(normal-cpu  PRIVATE -Wall -Wextra -Wpedantic -O3)