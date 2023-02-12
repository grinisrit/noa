set( tnl_mean_curvature_flow_SOURCES
     tnl-mean-curvature-flow.cpp
     tnl-mean-curvature-flow-eoc.cpp
     tnl-mean-curvature-flow.cu
     tnl-mean-curvature-flow-eoc.cu )

IF( TNL_BUILD_CUDA )
   CUDA_ADD_EXECUTABLE(tnl-mean-curvature-flow tnl-mean-curvature-flow.cu)
   CUDA_ADD_EXECUTABLE(tnl-mean-curvature-flow-eoc-test tnl-mean-curvature-flow-eoc.cu)
   target_link_libraries (tnl-mean-curvature-flow tnl  ${CUSPARSE_LIBRARY} )
   target_link_libraries (tnl-mean-curvature-flow-eoc-test tnl  ${CUSPARSE_LIBRARY} )
ELSE()
   ADD_EXECUTABLE(tnl-mean-curvature-flow tnl-mean-curvature-flow.cpp)
   ADD_EXECUTABLE(tnl-mean-curvature-flow-eoc-test tnl-mean-curvature-flow-eoc.cpp)
   target_link_libraries (tnl-mean-curvature-flow tnl )
   target_link_libraries (tnl-mean-curvature-flow-eoc-test tnl )
ENDIF()

INSTALL( TARGETS tnl-mean-curvature-flow
                 tnl-mean-curvature-flow-eoc-test
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES tnl-run-mean-curvature-flow-eoc-test
               tnl-run-mean-curvature-flow
               tnl-run-mean-curvature-flow-contour-video
               tnl-run-mean-curvature-flow-videos
               ${tnl_mean_curvature_flow_SOURCES}
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}/examples/mean-curvature-flow )
