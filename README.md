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
To load the nequip model `water-deploy.pth` using Fortran-C/C++ interface with the PyTorch C++ Frontend simply do: 
```
cd neural_net
./build/call_nequip_interface_fort
```
Instead, to load the nequip model `water-deploy.pth` using just the C/C++ interface with the PyTorch C++ Frontend simply do: 
```
cd neural_net
./build/call_nequip_interface
```

### Brief description:
* `fortran_call.90` is the main program, where coordinates and species information are defined, and where the `water-deploy.pth` model is declared and passed to the C/C++ code
* `wrap_nequip.f90` contains the Fortran module that is interfaced to C via `ISO_C_BINDING`
* `nequip_wrapper.cpp` and `nequip_wrapper.h` are the C/C++ code and header file, respectively, where the Pytorch model is loaded and where inference will eventually be made. The header file contains the `__cplusplus` preprocessor macro and the `extern "C"` function such that the C++ code has C linkage.

### TODO:
* write minimal neighbor lists subroutine in Fortran and then pass this list to the C/C++ code to be finally able to perform inference using the `compute_nequip` function inside nequip_wrapper.cpp. For now the module `neighnborlists.f90` only computes distances. This is how it would be done once Nequip is implemented in CP2K.
