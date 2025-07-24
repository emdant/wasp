cmake_minimum_required(VERSION 3.18)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(wasp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

add_compile_options("-Wno-interference-size")
add_compile_options("-march=native")

link_libraries("-lnuma")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(COMMAND gcc -print-search-dirs
    OUTPUT_VARIABLE gcc_install_dir
    RESULT_VARIABLE _result)
    if (NOT _result EQUAL "0")
    message(FATAL_ERROR "Failed fetching install path from gcc ${_result}")
  endif()

  string(REGEX MATCH "^install:[ \t]*([^\r\n]*)" gcc_install_dir "${gcc_install_dir}" )
  set(gcc_install_dir "${CMAKE_MATCH_1}")
  add_compile_options("--gcc-install-dir=${gcc_install_dir}")
endif()

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

include_directories(include)

option(LINK_PAPI "Link the PAPI library and compile the programs using PAPI" OFF)
if (LINK_PAPI)
  if(DEFINED ENV{PAPI_DIR})
    set(PAPI_DIR $ENV{PAPI_DIR})
    
    include_directories(${PAPI_DIR}/include)
    link_directories(${PAPI_DIR}/lib)
    link_libraries("papi")

    add_compile_definitions(PAPI_PROFILE)
  else()
    message(FATAL_ERROR "PAPI_DIR is not defined")
  endif()
endif()

option(COUNT_RELAX "Count the number of relaxations during SSSP" OFF)
option(COMPILE_UTILS "Enable the compilation of utilities" OFF)

set(TYPES int32 float)

function(create_typed_target TARGET_NAME SOURCE_FILE)
  add_custom_target(${TARGET_NAME})

  foreach(TYPE ${TYPES})
    set(TYPED_TARGET ${TARGET_NAME}-${TYPE})
    
    add_executable(${TYPED_TARGET} ${SOURCE_FILE})
    string(TOUPPER ${TYPE} UPPER_TYPE)
    target_compile_definitions(${TYPED_TARGET} PRIVATE USE_${UPPER_TYPE})
    
    add_dependencies(${TARGET_NAME} ${TYPED_TARGET})
  endforeach()
endfunction()

if (COUNT_RELAX)
  create_typed_target(sssp-crelax src/sssp.cc)
  foreach(TYPE ${TYPES})
    target_compile_definitions(sssp-crelax-${TYPE} PRIVATE COUNT_RELAX)
  endforeach()
endif()

create_typed_target(dijkstra src/dijkstra.cc)
create_typed_target(sssp src/sssp.cc)
# add_executable(bfs src/bfs.cc)

if (COMPILE_UTILS)
create_typed_target(converter utils/converter.cc)
create_typed_target(weight utils/weight.cc)
create_typed_target(degree utils/degree.cc)
create_typed_target(check-graph utils/check-graph.cc)
endif()

option(COMPILE_TESTS "Enable the compilation of tests" OFF)
if (COMPILE_TESTS)
  add_subdirectory(tests)
endif()