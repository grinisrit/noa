set( tnl_transport_equation_SOURCES
     tnl-transport-equation.cpp
     tnl-transport-equation.cu )

set( tnl_transport_equation_HEADERS
     tnl-transport-equation.h
     transportEquationBuildConfigTag.h
     transportEquationProblem.h
     transportEquationProblem_impl.h )


IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE( tnl-transport-equation tnl-transport-equation.cu)
   target_link_libraries( tnl-transport-equation ${CUSPARSE_LIBRARY} )
   CUDA_ADD_EXECUTABLE( tnl-transport-equation-eoc tnl-transport-equation-eoc.cu)
   target_link_libraries( tnl-transport-equation-eoc ${CUSPARSE_LIBRARY} )

ELSE()
   ADD_EXECUTABLE( tnl-transport-equation tnl-transport-equation.cpp)
   ADD_EXECUTABLE( tnl-transport-equation-eoc tnl-transport-equation-eoc.cpp)
ENDIF()


INSTALL( TARGETS tnl-transport-equation
                 tnl-transport-equation-eoc
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES tnl-run-transport-equation
               ${tnl_transport_equation_SOURCES}
               ${tnl_transport_equation_HEADERS}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/transport-equation )
