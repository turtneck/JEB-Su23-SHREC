cmake_minimum_required(VERSION 3.5)
project(bonus)

# require a C++11 compiler for all targets
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE test_database.cpp Database.hpp Database.tpp binary_search_tree.hpp binary_search_tree.tpp abstract_list.hpp list.hpp list.tpp)

add_executable(test-database test_database.cpp Database.hpp)

enable_testing()

add_test(test-database test-database)

add_custom_target(submission COMMAND
    ${CMAKE_COMMAND} -E tar "cfv" "${CMAKE_CURRENT_BINARY_DIR}/project_bonus.zip" --format=zip
       ${SOURCE}
       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  