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


add_library(wasp INTERFACE)
target_include_directories(wasp INTERFACE "include")
target_include_directories(wasp INTERFACE "extern")
target_compile_options(wasp INTERFACE -Wall -Wextra -Wpedantic -Wno-interference-size -Wshadow -Wnull-dereference -Wduplicated-branches -Wduplicated-cond -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual -Wnull-dereference)
target_compile_options(wasp INTERFACE "-march=native")
target_link_libraries(wasp INTERFACE "numa")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O2 -g -fsanitize=address -fno-omit-frame-pointer")


find_package(OpenMP REQUIRED)
target_link_libraries(wasp INTERFACE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(COMMAND gcc -print-search-dirs
    OUTPUT_VARIABLE gcc_install_dir
    RESULT_VARIABLE _result)
    if (NOT _result EQUAL "0")
    message(FATAL_ERROR "Failed fetching install path from gcc ${_result}")
  endif()

  string(REGEX MATCH "^install:[ \t]*([^\r\n]*)" gcc_install_dir "${gcc_install_dir}" )
  set(gcc_install_dir "${CMAKE_MATCH_1}")
  target_compile_options(wasp INTERFACE "--gcc-install-dir=${gcc_install_dir}")
endif()

option(PROFILE_STATS "Enable the profiling of statistics at runtime" OFF)

option(LINK_PAPI "Link the PAPI library and compile the programs using PAPI" OFF)
if (LINK_PAPI)
  if(DEFINED ENV{PAPI_DIR})
    set(PAPI_DIR $ENV{PAPI_DIR})
    
    target_include_directories(wasp INTERFACE ${PAPI_DIR}/include)
    target_link_directories(wasp INTERFACE ${PAPI_DIR}/lib)
    target_link_libraries(wasp INTERFACE "papi")
    target_compile_definitions(wasp INTERFACE PAPI_PROFILE)
  else()
    message(FATAL_ERROR "PAPI_DIR is not defined")
  endif()
endif()

### Variable storing the list of types to compile for
set(TYPES int32 float)

### Utility function to automatically create different executables for different types
function(create_typed_target TARGET_NAME SOURCE_FILE)
  add_custom_target(${TARGET_NAME})

  foreach(TYPE ${TYPES})
    set(TYPED_TARGET ${TARGET_NAME}-${TYPE})
    
    add_executable(${TYPED_TARGET} ${SOURCE_FILE})
    string(TOUPPER ${TYPE} UPPER_TYPE)
    target_compile_definitions(${TYPED_TARGET} PRIVATE USE_${UPPER_TYPE})
    target_link_libraries(${TYPED_TARGET} PRIVATE wasp)
    add_dependencies(${TARGET_NAME} ${TYPED_TARGET})
    
    if (PROFILE_STATS)
      target_compile_definitions(${TYPED_TARGET} PRIVATE PROFILE_STATS)
    endif()

  endforeach()
endfunction()

option(COUNT_RELAX "Count the number of relaxations during SSSP" OFF)
add_subdirectory(src)

option(COMPILE_UTILS "Enable the compilation of utilities" OFF)
if (COMPILE_UTILS)
  add_subdirectory(utils)
endif()

option(COMPILE_TESTS "Enable the compilation of tests" OFF)
if (COMPILE_TESTS)
  add_subdirectory(tests)
endif()