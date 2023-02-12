set( tnl_flow_sw_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_sw_SOURCES
     navierStokes.cpp
     navierStokes.cu )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-sw navierStokes.cu)
   target_link_libraries (tnl-navier-stokes-sw ${CUSPARSE_LIBRARY} )
ELSE()
   ADD_EXECUTABLE(tnl-navier-stokes-sw navierStokes.cpp)
ENDIF()


INSTALL( TARGETS tnl-navier-stokes-sw
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES run-navier-stokes-sw
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/navier-stokes-sw )
