###This CMakeLists.txt file is for call_nequip_interface 

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(nequip-fortran-c-interface)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(call_nequip_interface call_nequip_interface.c c_wrapper.cpp nequip.cpp)

target_link_libraries(call_nequip_interface "${TORCH_LIBRARIES}")

set_property(TARGET call_nequip_interface PROPERTY CXX_STANDARD 14)

#add_definitions(-DNEQUIP)
