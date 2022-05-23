set( tnl_heat_equation_SOURCES     
     euler.cpp
     euler-cuda.cu )
               
IF( BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-euler-3d${debugExt} euler-cuda.cu)
   target_link_libraries (tnl-euler-3d${debugExt} tnl${debugExt}-${tnlVersion}  ${CUSPARSE_LIBRARY} )
ELSE(  BUILD_CUDA )               
   ADD_EXECUTABLE(tnl-euler-3d${debugExt} euler.cpp)     
   target_link_libraries (tnl-euler-3d${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF( BUILD_CUDA )


INSTALL( TARGETS tnl-euler-3d${debugExt}
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
        
INSTALL( FILES run-euler
               ${tnl_heat_equation_SOURCES}
         DESTINATION share/tnl-${tnlVersion}/examples/inviscid-flow-3d )

