install( DIRECTORY async mpark Leksys TYPE INCLUDE
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

if( ${BUILD_PYTHON} )
   install( DIRECTORY cctbx TYPE INCLUDE
            MESSAGE_NEVER
            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )
endif()
