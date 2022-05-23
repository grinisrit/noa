set( tnl_inviscid_flow_sw_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_inviscid_flow_sw_SOURCES     
     euler.cpp
     euler.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-euler-sw euler.cu)
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-euler-sw${debugExt} euler.cpp)     
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-euler-sw
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-euler-sw
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/inviscid-flow-sw )
