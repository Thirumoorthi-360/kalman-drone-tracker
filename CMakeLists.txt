cmake_minimum_required(VERSION 3.16)
project(kalman_drone_tracker VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(KF_BUILD_TESTS "Build unit tests" ON)

# ---- library ----
add_library(kftrack
  src/hungarian.cpp
  src/tracker.cpp
  src/simulation.cpp
  src/metrics.cpp
)
target_include_directories(kftrack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(kftrack PUBLIC /W4 /utf-8)
else()
  target_compile_options(kftrack PUBLIC -Wall -Wextra -Wpedantic)
endif()

# ---- programs ----
add_executable(drone_tracker apps/drone_tracker.cpp)
target_link_libraries(drone_tracker PRIVATE kftrack)

add_executable(gps_car_1d examples/gps_car_1d.cpp)
target_link_libraries(gps_car_1d PRIVATE kftrack)

# ---- tests ----
if(KF_BUILD_TESTS)
  enable_testing()
  add_executable(kf_tests tests/test_main.cpp)
  target_link_libraries(kf_tests PRIVATE kftrack)
  add_test(NAME kf_tests COMMAND kf_tests)
endif()
