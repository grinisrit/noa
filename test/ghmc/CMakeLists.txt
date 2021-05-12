####################################################################################################
##################                    GHMC functional tests                    #####################
####################################################################################################


# Main executables
add_executable(sample_normal sample-normal.cc)
add_executable(sample_funnel sample-funnel.cc)


# Link libraries
target_link_libraries(sample_normal PRIVATE ${PROJECT_NAME})
target_compile_options(sample_normal PRIVATE -Wall -Wextra -Wpedantic -O3)

target_link_libraries(sample_funnel PRIVATE ${PROJECT_NAME})
target_compile_options(sample_funnel PRIVATE -Wall -Wextra -Wpedantic -O3)