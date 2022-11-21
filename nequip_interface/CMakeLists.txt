###This CMakeLists.txt file is for call_nequip_interface 

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(nequip-fortran-c-interface)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_Fortran_COMPILER  gfortran)
ENABLE_LANGUAGE(Fortran)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_Fortran_FLAGS "-lstdc++")

add_executable(call_nequip_interface call_nequip_interface.c nequip_wrapper.cpp nequip.cpp)
target_link_libraries(call_nequip_interface "${TORCH_LIBRARIES}")
set_property(TARGET call_nequip_interface PROPERTY CXX_STANDARD 14)

add_executable(call_nequip_interface_fort fortran_call.f90 wrap_nequip.f90 nequip_wrapper.cpp nequip.cpp)
target_link_libraries(call_nequip_interface_fort "${TORCH_LIBRARIES}")
set_property(TARGET call_nequip_interface_fort PROPERTY CXX_STANDARD 14)
set_property(TARGET call_nequip_interface_fort PROPERTY LINKER_LANGUAGE Fortran)


#add_definitions(-DNEQUIP)
