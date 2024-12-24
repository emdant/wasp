cmake_minimum_required(VERSION 3.18)

project(wasp)
set (CMAKE_CXX_STANDARD 17)

add_compile_options("-Wno-interference-size")
add_compile_options("-march=native")

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

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  link_libraries(OpenMP::OpenMP_CXX)
endif()

include_directories(include)

option(LINK_PAPI "Link the PAPI library and compile the programs using PAPI" OFF)
if (LINK_PAPI)
  if(DEFINED ENV{PAPI_DIR})
    set(PAPI_DIR $ENV{PAPI_DIR})
    
    include_directories(${PAPI_DIR}/include)
    link_directories(${PAPI_DIR}/lib)
    link_libraries("papi")

    add_compile_definitions(PAPI_PROFILE)

  endif()
endif()


add_executable(dijkstra src/dijkstra.cc)
add_executable(sssp src/sssp.cc)
add_executable(bfs src/bfs.cc)

add_executable(converter utils/converter.cc)
add_executable(stats utils/stats.cc)
add_executable(weight utils/weight.cc)
add_executable(degree utils/degree.cc)

add_subdirectory(tests)