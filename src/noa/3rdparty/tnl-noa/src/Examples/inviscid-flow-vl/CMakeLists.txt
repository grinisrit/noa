set( tnl_inviscid_flow_vl_HEADERS
     CompressibleConservativeVariables.h )

set( tnl_inviscid_flow_vl_SOURCES
     euler.cpp
     euler.cu )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-euler-vl euler.cu)
ELSE()
   ADD_EXECUTABLE(tnl-euler-vl euler.cpp)
ENDIF()


INSTALL( TARGETS tnl-euler-vl
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES run-euler-vl
               ${tnl_inviscid_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/inviscid-flow-vl )
