INSTALL( FILES tnl-quickstart
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES build-config-tag.h.in
               Makefile
               config.mk
               explicit-laplace-grid-1d_impl.h.in
               explicit-laplace-grid-2d_impl.h.in
               explicit-laplace-grid-3d_impl.h.in
               implicit-laplace-grid-1d_impl.h.in
               implicit-laplace-grid-2d_impl.h.in
               implicit-laplace-grid-3d_impl.h.in
               main.h.in
               main.cpp.in
               main.cu.in
               operator.h.in
               operator_impl.h.in
               operator-grid-specialization.h.in
               operator-grid-specialization_impl.h.in
               problem.h.in
               problem_impl.h.in
               rhs.h.in
               run-script.in
         DESTINATION ${TNL_TARGET_DATA_DIRECTORY}
         PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ )
