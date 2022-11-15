#!/bin/bash -e

DOWNLOAD_LIBTORCH=true

if [ "$DOWNLOAD_LIBTORCH" = true ]
then

echo "Installing requirements (pytorch 1.13)"
rm -f libtorch-shared-with-deps-latest.zip 
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip -o libtorch-shared-with-deps-latest.zip
echo "Build interface"
cd nequip_interface && mkdir -p build \
&& cd build && cmake  -DCMAKE_PREFIX_PATH=/data/gtocci/scratch/projects/nequip-C-fortran-interface/libtorch .. \
&& cmake --build . --config Release

else
echo "Installing requirements (pytorch 1.13)"
pip3 install -r requirements.txt
echo "Build interface"
cd nequip_interface && mkdir -p build \
&& cd build && cmake  -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` .. \
&& cmake --build . --config Release
fi
