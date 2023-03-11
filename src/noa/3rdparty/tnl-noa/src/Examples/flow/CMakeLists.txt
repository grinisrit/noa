set( tnl_flow_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_SOURCES
     navierStokes.cpp
     navierStokes.cu )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes navierStokes.cu)
ELSE()
   ADD_EXECUTABLE(tnl-navier-stokes navierStokes.cpp)
ENDIF()


INSTALL( TARGETS tnl-navier-stokes
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES run-navier-stokes
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/flow )
