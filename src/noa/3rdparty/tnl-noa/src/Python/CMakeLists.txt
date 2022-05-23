find_package( PythonInterp 3 )
find_package( PythonLibs 3 )

set( PYTHON_SITE_PACKAGES_DIR lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages )

if( PYTHONINTERP_FOUND )
   CONFIGURE_FILE( "__init__.py.in" "__init__.py" )
   INSTALL( FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
                  BenchmarkLogs.py
                  BenchmarkPlots.py
                  LogParser.py
            DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/TNL )
endif()

if( PYTHONLIBS_FOUND )
   # check if pybind11 exists on the system
   find_package(pybind11 QUIET)
   if( NOT pybind11_FOUND )
      # download and build pybind11 at configure time
      configure_file(pybind11.cmake.in ${CMAKE_BINARY_DIR}/pybind11-download/CMakeLists.txt)
      execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
         RESULT_VARIABLE result
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pybind11-download )
      if(result)
         message(FATAL_ERROR "CMake step for pybind11 failed: ${result}")
      endif()
      execute_process(COMMAND ${CMAKE_COMMAND} --build .
         RESULT_VARIABLE result
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pybind11-download )
      if(result)
         message(FATAL_ERROR "Build step for pybind11 failed: ${result}")
      endif()

      # add the pybind11 subdirectory to provide the pybind11_add_module macro
      add_subdirectory(${CMAKE_BINARY_DIR}/pybind11-src ${CMAKE_BINARY_DIR}/pybind11-build)
   endif()

   # add the subdirectory with our bindings
   add_subdirectory(pytnl)
else()
   message( "The Python.h header file was not found, Python bindings will not be builg." )
endif()
