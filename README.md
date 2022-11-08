# NequIP-C/C++-Fortran-interface
Fortran - C/C++ interface for NequIP.

Interface between Fortran, C and the [C++ frontend of Pytorch](https://pytorch.org/tutorials/advanced/cpp_frontend.html) for the [NequIP](https://github.com/mir-group/nequip) code. 

The interface is built following the [DeePMD-kit-Fortran-Cpp-interface](https://github.com/Cloudac7/DeePMD-kit-Fortran-Cpp-interface) as well as the LAMMPS interface [pair_nequip interface](https://github.com/mir-group/pair_nequip). The plan is to merge the interface to [CP2K](https://github.com/cp2k/cp2k).

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
### Running the code
To perform inference with nequip on a 32 water molecules box using the model `water-deploy.pth` with the Fortran-C/C++ interface and the PyTorch C++ Frontend simply do: 
```
cd nequip_interface
./build/call_nequip_interface_fort
```
Instead, to do it just with the C/C++ interface and Pytorch C++ Frontend do: 
```
cd nequip_interface
./build/call_nequip_interface
```

### Brief description:
* `fortran_call.90` is the main program, where coordinates and species information are defined, and where the `water-deploy.pth` model is declared and passed to C/C++
* `wrap_nequip.f90` contains the Fortran module that is interfaced to C via `ISO_C_BINDING`
* `c_wrapper.cpp` and `c_wrapper.h` are the C/C++ code and header, respectively, for interoperability between C and C++. The header file contains the `__cplusplus` preprocessor macro and the `extern "C"` function such that the C++ code has C linkage. `c_wrapper.cpp` calls `nequip.cpp`
* `nequip.h` and `nequip.cpp` are the C++ code and header containing the `NequipPot` class that takes care of loading the model and of inference.

### TODO:
* write minimal neighbor lists subroutine in Fortran and then pass this list to the C/C++ code to be finally able to perform inference using the `compute_nequip` function inside nequip_wrapper.cpp. For now the module `neighnborlists.f90` only computes distances. This is how it would be done once Nequip is implemented in CP2K.

### Note:
* Right now I have bypassed the calculation with neighborlist, as the only thing that is required is to do a double loop over atoms to calculate neighbors within a cutoff to get the indixes of the connected atom pairs (edges) and the cell_shifts identifying whether the connection from first to second crosses a periodic boundary. 