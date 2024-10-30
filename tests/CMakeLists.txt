add_executable(all_tests 
  main.cc 
  deque_test.cc
  )
target_include_directories(all_tests PRIVATE .)
set_target_properties(all_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
