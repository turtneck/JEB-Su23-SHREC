cmake_minimum_required(VERSION 3.5)
project(P5 CXX)

# require a C++11 compiler for all targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE test_frontier_queue.cpp test_solver.cpp frontier_queue.hpp frontier_queue.tpp)
  
add_executable(test-frontier-queue test_frontier_queue.cpp catch.hpp
  state.hpp state.tpp
  frontier_queue.hpp frontier_queue.tpp)

add_executable(test-solver test_solver.cpp catch.hpp
  puzzle.hpp puzzle.cpp
  state.hpp state.tpp
  frontier_queue.hpp frontier_queue.tpp
  puzzle_solver.hpp puzzle_solver.cpp)

enable_testing()
add_test(test-frontier test-frontier-queue)
add_test(test-solver test-solver)

add_custom_target(submission COMMAND
  ${CMAKE_COMMAND} -E tar "cfv" "${CMAKE_CURRENT_BINARY_DIR}/p5_puzzle_solver.zip" --format=zip
  ${SOURCE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
