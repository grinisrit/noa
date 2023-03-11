#ADD_EXECUTABLE(tnl-hamilton-jacobi main.cpp)

IF( TNL_BUILD_CUDA )
    CUDA_ADD_EXECUTABLE(tnl-direct-eikonal-solver tnl-direct-eikonal-solver.cu )
ELSE()
    ADD_EXECUTABLE(tnl-direct-eikonal-solver tnl-direct-eikonal-solver.cpp )
ENDIF()


INSTALL( TARGETS #tnl-hamilton-jacobi
                 tnl-direct-eikonal-solver
         RUNTIME DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )
