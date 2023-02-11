set( tnl_heat_equation_SOURCES
     tnl-heat-equation.cpp
     tnl-heat-equation-eoc.cpp
     tnl-heat-equation.cu
     tnl-heat-equation-eoc.cu )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-heat-equation tnl-heat-equation.cu)
   CUDA_ADD_EXECUTABLE(tnl-heat-equation-eoc-test tnl-heat-equation-eoc.cu)
   target_link_libraries (tnl-heat-equation ${CUSPARSE_LIBRARY} )
   target_link_libraries (tnl-heat-equation-eoc-test ${CUSPARSE_LIBRARY} )
ELSE()
   ADD_EXECUTABLE(tnl-heat-equation tnl-heat-equation.cpp)
   ADD_EXECUTABLE(tnl-heat-equation-eoc-test tnl-heat-equation-eoc.cpp)
ENDIF()


INSTALL( TARGETS tnl-heat-equation
                 tnl-heat-equation-eoc-test
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES tnl-run-heat-equation-eoc-test
               tnl-run-heat-equation
               ${tnl_heat_equation_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/heat-equation )
