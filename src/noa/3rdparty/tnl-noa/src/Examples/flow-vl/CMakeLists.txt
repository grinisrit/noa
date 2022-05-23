set( tnl_flow_vl_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_flow_vl_SOURCES     
     navierStokes.cpp
     navierStokes.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-navier-stokes-vl navierStokes.cu)
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-navier-stokes-vl navierStokes.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-navier-stokes-vl
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-navier-stokes-vl
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/navier-stokes-vl )
