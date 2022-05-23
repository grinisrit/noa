set( HOST_EXAMPLES
   StaticODESolver-SineExample
   StaticODESolver-LorenzExample
)

set( COMMON_EXAMPLES
   StaticODESolver-SineParallelExample
   StaticODESolver-LorenzParallelExample
   ODESolver-HeatEquationExample
   ODESolver-HeatEquationWithMonitorExample
)

foreach( target IN ITEMS ${HOST_EXAMPLES})
   add_executable( ${target} ${target}.cpp )
   add_custom_command( COMMAND ${target} > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
      OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
             ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}-result.out )
   set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
endforeach()

if( BUILD_CUDA )
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      cuda_add_executable( ${target}-cuda ${target}.cu OPTIONS )
      add_custom_command( COMMAND ${target}-cuda ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}
         > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
         OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}-result.out)
      set( CUDA_OUTPUTS ${CUDA_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
else()
   foreach( target IN ITEMS ${COMMON_EXAMPLES} )
      add_executable( ${target} ${target}.cpp )
      add_custom_command( COMMAND ${target} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}
         > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
         OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out
                ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}-result.out )
      set( HOST_OUTPUTS ${HOST_OUTPUTS} ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/${target}.out )
   endforeach()
endif()

IF( BUILD_CUDA )
   ADD_CUSTOM_TARGET( RunODESolversExamples-cuda ALL DEPENDS ${CUDA_OUTPUTS} )
ELSE()
   ADD_CUSTOM_TARGET( RunODESolversExamples ALL DEPENDS ${HOST_OUTPUTS} )
ENDIF()

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.png
   OUTPUT StaticODESolver-SineExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineExample.out
   VERBATIM )

   ADD_CUSTOM_COMMAND(
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-SineParallelExample.py
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample-result.out
         ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.png
      OUTPUT StaticODESolver-SineParallelExample.png
      DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-SineParallelExample.out
      VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.png
   OUTPUT StaticODESolver-LorenzExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/StaticODESolver-LorenzParallelExample.py
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample-result.out
      ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample
   OUTPUT StaticODESolver-LorenzParallelExample.png
   DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/StaticODESolver-LorenzParallelExample.out
   VERBATIM )

ADD_CUSTOM_COMMAND(
COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/ODESolver-HeatEquationExample.py
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample-result.out
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample
OUTPUT ODESolver-HeatEquationExample.png
DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/ODESolver-HeatEquationExample.out
VERBATIM )


ADD_CUSTOM_TARGET( ProcessODESolversExamplesResults ALL DEPENDS
      StaticODESolver-SineExample.png
      StaticODESolver-SineParallelExample.png
      StaticODESolver-LorenzExample.png
      StaticODESolver-LorenzParallelExample.png
      ODESolver-HeatEquationExample.png )
