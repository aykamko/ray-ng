cmake_minimum_required(VERSION 2.8)

project(photon)

if(NOT APPLE)
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)
  set(CUSTOM_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
else()
  find_program(CUSTOM_PYTHON_EXECUTABLE python)
  message("-- Found Python program: ${CUSTOM_PYTHON_EXECUTABLE}")
  execute_process(COMMAND ${CUSTOM_PYTHON_EXECUTABLE} -c
                    "import sys; print 'python' + sys.version[0:3]"
                    OUTPUT_VARIABLE PYTHON_LIBRARY_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${CUSTOM_PYTHON_EXECUTABLE} -c
                    "import sys; print sys.exec_prefix"
                    OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    FIND_LIBRARY(PYTHON_LIBRARIES
                 NAMES ${PYTHON_LIBRARY_NAME}
                 HINTS "${PYTHON_PREFIX}"
                 PATH_SUFFIXES "lib" "libs"
                 NO_DEFAULT_PATH)
    execute_process(COMMAND ${CUSTOM_PYTHON_EXECUTABLE} -c
                    "from distutils.sysconfig import *; print get_python_inc()"
                    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYTHON_LIBRARIES AND PYTHON_INCLUDE_DIRS)
        SET(PYTHONLIBS_FOUND TRUE)
        message("-- Found PythonLibs: " ${PYTHON_LIBRARIES})
        message("-- -- Used custom search path")
    else()
        find_package(PythonLibs REQUIRED)
        message("-- -- Used find_package(PythonLibs)")
    endif()
endif()

if(APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

include_directories("${PYTHON_INCLUDE_DIRS}")

set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} --std=c99 -Werror")

if (UNIX AND NOT APPLE)
   link_libraries(rt)
endif()

set(PHOTON_CLIENT_LIB "${CMAKE_SOURCE_DIR}/build/photon_client.a" CACHE STRING
  "Path to photon_client.a")

set(COMMON_LIB "${CMAKE_SOURCE_DIR}/../common/build/libcommon.a" CACHE STRING
  "Path to libcommon.a")

include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${CMAKE_SOURCE_DIR}/../")
include_directories("${CMAKE_SOURCE_DIR}/../common/")
include_directories("${CMAKE_SOURCE_DIR}/../common/thirdparty/")
include_directories("${CMAKE_SOURCE_DIR}/../common/lib/python/")

add_library(photon SHARED
  photon_extension.c
  ../common/lib/python/common_extension.c)

get_filename_component(PYTHON_SHARED_LIBRARY ${PYTHON_LIBRARIES} NAME)
if(APPLE)
  add_custom_command(TARGET photon
      POST_BUILD COMMAND
      ${CMAKE_INSTALL_NAME_TOOL} -change ${PYTHON_SHARED_LIBRARY} ${PYTHON_LIBRARIES} libphoton.so)
endif(APPLE)

target_link_libraries(photon ${PHOTON_CLIENT_LIB} ${COMMON_LIB} ${PYTHON_LIBRARIES})

install(TARGETS photon DESTINATION ${CMAKE_SOURCE_DIR}/photon)