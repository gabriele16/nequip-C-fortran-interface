cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(neural_net neural_net.cpp)
add_executable(neural_net_submod neural_net_submod.cpp)
add_executable(neural_net_load neural_net_load.cpp)
add_executable(load_nequip load_nequip.cpp)

target_link_libraries(neural_net "${TORCH_LIBRARIES}")
target_link_libraries(neural_net_submod "${TORCH_LIBRARIES}")
target_link_libraries(neural_net_load "${TORCH_LIBRARIES}")
target_link_libraries(load_nequip "${TORCH_LIBRARIES}")

set_property(TARGET neural_net PROPERTY CXX_STANDARD 14)
set_property(TARGET neural_net_submod PROPERTY CXX_STANDARD 14)
set_property(TARGET neural_net_load PROPERTY CXX_STANDARD 14)
set_property(TARGET load_nequip PROPERTY CXX_STANDARD 14)
