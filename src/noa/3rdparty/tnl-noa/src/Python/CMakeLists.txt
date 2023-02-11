find_package( PythonInterp 3 )

set( PYTHON_SITE_PACKAGES_DIR lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages )

if( PYTHONINTERP_FOUND )
   CONFIGURE_FILE( "__init__.py.in" "__init__.py" )
   INSTALL( FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
                  BenchmarkLogs.py
                  BenchmarkPlots.py
                  LogParser.py
            DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/TNL )
endif()
