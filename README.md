# CUBIC-informatics

## Getting Started

### Prerequisites
 * Cmake 2.8
 * GNU Compiler Collection 4.8
 * Python 3.6
 * [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-toolkit)
 * [ANTs 2.1.0](https://github.com/ANTsX/ANTs/releases)

### Downloading source codes
```
git clone https://github.com/lsb-riken/CUBIC-informatics
cd CUBIC-informatics
```

### Building C++/CUDA programs
Some programs use helper functions in CUDA Toolkit. Create a symbolic link in the source directory. The path of CUDA Toolkit depends on your system setup.

```
ln -s /usr/local/cuda/samples/common common
```

Build programs under `build` directory. Please change `CUDA_NVCC_FLAGS` in `CMakeLists.txt` to match the [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU devices.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

### Preparing parameter files


### Running programs
