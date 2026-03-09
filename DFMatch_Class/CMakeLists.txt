cmake_minimum_required(VERSION 3.0.2)
#SET(CMAKE_C_COMPILER /home/bin.fei/work/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
#SET(CMAKE_CXX_COMPILER /home/bin.fei/work/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
set(CMAKE_CXX_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
project(DFMatch)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

include_directories(include)
link_directories(lib)
add_library(DFMatch_lib
  src/DFeat.cpp
)
add_executable(DFMatch main.cpp)

target_link_libraries(DFMatch DFMatch_lib hbmem alog opencv_world dnn hbrt_bayes_aarch64 cnn_intf)
