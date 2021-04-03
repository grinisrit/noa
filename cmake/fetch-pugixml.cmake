FetchContent_Declare(
    pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG        v1.11.4
  )

FetchContent_GetProperties(pugixml)

if(NOT pugixml_POPULATED)
    FetchContent_Populate(pugixml)
    set(PUGIXML_SRC "${pugixml_SOURCE_DIR}/src")
endif()