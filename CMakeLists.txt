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
if (COUNT_RELAX)
  add_executable(sssp-crelax src/sssp.cc)
  target_compile_definitions(sssp-crelax PRIVATE COUNT_RELAX)
endif()

add_executable(dijkstra src/dijkstra.cc)
add_executable(sssp src/sssp.cc)
# add_executable(bfs src/bfs.cc)

option(COMPILE_UTILS "Enable the compilation of utilities" OFF)
if (COMPILE_UTILS)
add_executable(converter utils/converter.cc)
add_executable(weight utils/weight.cc)
add_executable(degree utils/degree.cc)
add_executable(check-graph utils/check-graph.cc)
endif()

option(COMPILE_TESTS "Enable the compilation of tests" OFF)
if (COMPILE_TESTS)
  add_subdirectory(tests)
endif()