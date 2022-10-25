cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nequip-fortran-c-interface)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)

SET(CMAKE_Fortran_COMPILER  gfortran)

ENABLE_LANGUAGE(Fortran)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_Fortran_FLAGS "-lstdc++")

add_executable(call_nequip_c call_nequip.c nequip_wrapper.cpp)
#add_executable(call_nequip_fort fortran_call.f90 wrap_nequip.f90 neighborlists.f90 nequip_wrapper.cpp )

#target_link_libraries(call_nequip_fort "${TORCH_LIBRARIES}")
target_link_libraries(call_nequip_c "${TORCH_LIBRARIES}")

#set_property(TARGET call_nequip_fort PROPERTY CXX_STANDARD 14)
#set_property(TARGET call_nequip_fort PROPERTY LINKER_LANGUAGE Fortran)
set_property(TARGET call_nequip_c PROPERTY CXX_STANDARD 14)

#add_definitions(-DNEQUIP)
