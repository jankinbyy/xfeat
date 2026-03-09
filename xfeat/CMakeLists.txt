cmake_minimum_required(VERSION 3.12)
project(xfeat_cpp)
set(CMAKE_BUILD_TYPE Debug)
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
# set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
include_directories(
    ${OpenCV_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDEDIR}
)
set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
add_subdirectory(src)
add_subdirectory(examples)
