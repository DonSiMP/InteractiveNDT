cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ndt)
set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(PCL 1.5 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(ndt pcl_demo.cpp)
target_link_libraries (ndt ${PCL_LIBRARIES})