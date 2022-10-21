# nequip-C-fortran-interface
Fortran - C/C++ interface for nequip.

Right now this code is following the tutorial for the [PYTORCH C++ FRONTEND](https://pytorch.org/tutorials/advanced/cpp_frontend.html) and it is taking as example the [DeePMD-kit-Fortran-Cpp-interface](https://github.com/Cloudac7/DeePMD-kit-Fortran-Cpp-interface) as well as the LAMMPS interface [pair_nequip interface](https://github.com/mir-group/pair_nequip).

### Installation
This interface currently requires:
* Python >= 3.7
* pip3
* cmake >= 3.0.0
* gfortran v>= 11.2.0
* gcc >= 11.2.0

The following installation script will run `pip3 install requirements.txt` and run `cmake`:
```
./install_interface.sh
```
### Basic Neural Network Calculation
To load the nequip model `water-deploy.pth` using Fortran-C interface with the PyTorch C++ Frontend simply do: 
```
cd neural_net
./build/neural_net
```
### TODO:
* write minimal neighbor lists subroutine in Fortran and then pass it to the C/C++ code to be finally able to perform inference using the `compute_nequip` function inside nequip_wrapper.cpp
