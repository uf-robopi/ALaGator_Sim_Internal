cmake_minimum_required(VERSION 3.0.2)
project(uw_acoustics)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  gazebo_ros
)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs gazebo_ros
  DEPENDS gazebo
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(gazebo_acoustic_source SHARED src/acoustic_source_plugin.cpp)
target_link_libraries(gazebo_acoustic_source ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(gazebo_hydrophone SHARED src/hydrophone_plugin.cpp)
target_link_libraries(gazebo_hydrophone ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

install(TARGETS gazebo_acoustic_source gazebo_hydrophone
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})




catkin_install_python(PROGRAMS
  src/tdoa_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/fdoa_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  src/fdoa_rrd_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


catkin_install_python(PROGRAMS
  src/ukf_localizer_tdoa_fdoa.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

