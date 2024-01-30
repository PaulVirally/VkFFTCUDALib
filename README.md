# VkFFTCUDALib
This is the C++ library for the [Julia bindings](https://github.com/PaulVirally/VkFFTCUDA.jl) to [VkFFT](https://github.com/DTolm/VkFFT). There is no need to use this repository unless you are trying to modify the C++ library itself. The Julia bindings can be found here: [VkFFTCUDA.jl](https://github.com/PaulVirally/VkFFTCUDA.jl)

## Building the library
If you need to build the library on its own for some reason (there is no reason to do this normally if you are just using [VkFFTCUDA.jl](https://github.com/PaulVirally/VkFFTCUDA.jl)), there is nothing terribly non-standard to be done:
```
git clone https://github.com/PaulVirally/VkFFTCUDALib/ --recurse-submodules
cd VkFFTCUDALib
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release
make
sudo make install
```
