# CUBIC-informatics

## Getting Started

### Prerequisites
Tested on Cent OS 7.5 with the following versions of software.

 * Cmake 2.8.12
 * GNU Compiler Collection 4.8.5
 * Python 3.6.4
 * Julia 0.6.3
 * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
 * [ANTs](https://github.com/ANTsX/ANTs/releases) 2.1.0

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
### Installing python packages
```
pip install -r requirements.txt
```

## Cell/Nucleus detection
1. Preparing _HDoG_ parameter files for nuclear staining images

    example: `param/param_example_HDoG_FW.json`, `param/param_example_HDoG_RV.json`

2. Preparing _Merge_ parameter file

    example: `param/param_example_mergebrain.json`

3. Preparing _HDoG_ and _Merge_ parameter file as well for other channel images

4. Preparing _MultiChannel_ parameter file

    example: `param/param_example_multichannel.json`

### Tiling check
```
python script/MergeBrain.py images param/param_example_mergebrain.json
```

### Cell candidate detection
```
python script/HDoG_gpu.py param/param_example_HDoG_FW.json
python script/HDoG_gpu.py param/param_example_HDoG_RV.json
python script/MergeBrain.py cells param/param_example_mergebrain.json
```

### Candidate verification


### Multi-channel verification
```
python script/MultiChannelVerification.py param/param_example_multichannel.json
```

## Alignment and annotation
1. Preparing _Mapping_ parameter file

    example: `param/param_example_mapping.json`

### Registration
```
python script/AtlasMapping.py registration param/param_example_mapping.json
```

### Annotation
```
python script/AtlasMapping.py annotation param/param_example_mapping.json
```
