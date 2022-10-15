find_package( Doxygen REQUIRED )
set( DOXYGEN_EXCLUDE_PATTERNS "*/3rdparty/*" )
set( DOXYGEN_OUTPUT_DIRECTORY "doxygen" )
doxygen_add_docs(	noa-docs
			"docs/" "src/" "test/"
			WORKING_DIRECTORY "${CMAKE_PROJECT_DIR}"
		)
