#!/bin/bash -e

echo "Installing requirements (pytorch 1.12)"
pip3 install -r requirements.txt
echo "Build interface"
cd neural_net && mkdir -p build \
&& cd build && cmake  -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. \
&& cmake --build . --config Release
