# nequip-C-fortran-interface
Fortran - C/C++ interface for nequip.

Right now this code is following step-by-step the tutorial for the [PYTORCH C++ FRONTEND](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
### Installation
This interface currently requires:
* Python >= 3.7
Run:
```
./install_interface.sh
```
### Basic Neural Network Calculation
To run the Neural Network using the PyTorch C++ Frontend simply do: 
```
neural_net/build/neural_net
```
or using submodules, as recommended in the official documentation:
```
neural_net/build/neural_net_submod
```
To run the corresponding python code do instead:
```
python3 neural_net/neural_net.py
```
