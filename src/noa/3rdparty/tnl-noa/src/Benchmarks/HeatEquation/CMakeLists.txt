IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cu )

    CUDA_ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation-bug tnl-benchmark-simple-heat-equation-bug.cu )

#    CUDA_ADD_EXECUTABLE( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cu )
ELSE()
#    ADD_EXECUTABLE( tnl-benchmark-heat-equation tnl-benchmark-heat-equation.cpp )

    ADD_EXECUTABLE( tnl-benchmark-simple-heat-equation tnl-benchmark-simple-heat-equation.cpp )
ENDIF()


INSTALL( TARGETS
#            tnl-benchmark-heat-equation
            tnl-benchmark-simple-heat-equation
         RUNTIME DESTINATION bin )

if( BUILD_CUDA )
    INSTALL( TARGETS
                tnl-benchmark-simple-heat-equation-bug
             RUNTIME DESTINATION bin )
endif()