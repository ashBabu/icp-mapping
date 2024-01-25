cmake_minimum_required(VERSION 3.0.2)
project(icp_mapping)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pcd_generator
#  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_ros roscpp rospy sensor_msgs
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp_mapping
  src/icp_mapping.cpp
) 
target_link_libraries(
  icp_mapping 
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PCL_COMMON}
)  
