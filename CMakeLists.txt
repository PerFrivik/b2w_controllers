cmake_minimum_required(VERSION 3.8)
project(b2w_controllers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
list(APPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_ROOT}")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# find_package(onnxruntime REQUIRED)  # Add this line to find onnxruntime
find_package(onnxruntime REQUIRED
  HINTS "${ONNXRUNTIME_ROOT}"
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(b2w_controllers src/b2w_controllers.cpp)

install(DIRECTORY include/
  DESTINATION include
)

set_target_properties(onnxruntime::onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# target_link_libraries(b2w_controllers /home/perfrivik/Downloads/onnxruntime-linux-x64-1.20.1/lib64/libonnxruntime.so)
# include_directories(/usr/local/onnxruntime/include)

target_link_libraries(b2w_controllers onnxruntime::onnxruntime)


ament_target_dependencies(b2w_controllers rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs onnxruntime tf2 tf2_ros tf2_geometry_msgs)  # Add onnxruntime here

# target_include_directories(b2w_controllers PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_include_directories(b2w_controllers PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/local/onnxruntime/include
)

target_compile_features(b2w_controllers PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS b2w_controllers
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
