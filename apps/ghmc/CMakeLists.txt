####################################################################################################
################                     NOA Apps: GHMC examples                    ##################
####################################################################################################

message(STATUS "Building GHMC apps required")

# Main executables
add_executable(normal-cpu normal-cpu.cc)

# Link libraries
target_link_libraries(normal-cpu PRIVATE ${PROJECT_NAME})
target_compile_options(normal-cpu PRIVATE -Wall -Wextra -Wpedantic -O3)
