####################################################################################################
#####################              NOA unit & functional testing             #######################
####################################################################################################

message(STATUS "Building tests required")


# Functional tests
add_subdirectory(ghmc)
add_subdirectory(pms)
add_subdirectory(cfd)
add_subdirectory(quant)

# Unit tests
add_subdirectory(unit)