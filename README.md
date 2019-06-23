# CUBIC-informatics

## Getting Started

### Hardware Prerequisites
 * One or more NVIDIA GPUs (tested on [compute capability](https://developer.nvidia.com/cuda-gpus) 6.1 and 5.2)

### Software Prerequisites
Tested on Cent OS 7.5 with the following versions of software.

 * Cmake 2.8.12
 * GNU Compiler Collection 4.8.5
 * Python 3.6.4
 * Julia 0.6.3
 * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
 * [ANTs](https://github.com/ANTsX/ANTs/releases) 2.1.0

### Source Image Prerequisites

1. Source images should be in binary format. Batch image conversion from TIFF to binary can be performed with `script/tiff2bin.py`.

2. Source image files are assumed to be named like:

  `/path/to/src/YNAME/YNAME_XNAME/ZNAME.bin`

 * `XNAME`,`YNAME` is the number specifies the origin of the stack
 * `ZNAME` is the slice number

Conversion rule from these names to physical position should be provided in the _HDoG_ parameter files.

### Downloading source codes and reference data
```
git clone https://github.com/lsb-riken/CUBIC-informatics
cd CUBIC-informatics
```

Download CUBIC-Atlas reference data from [here](http://cubic-atlas.riken.jp/).

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
Whole brain image in TIFF format is created.

### Cell candidate detection
```
python script/HDoG_gpu.py param/param_example_HDoG_FW.json
python script/HDoG_gpu.py param/param_example_HDoG_RV.json
```
Cell candidate detection is performed on GPUs for forward side and reverse side.

```
python script/MergeBrain.py cells param/param_example_mergebrain.json
```
Candidate detection results on each side is merged.

### Candidate verification
```
python script/HDoG_classify.py param/param_example_classify.json
```
Plot every candidate in feature space and unsupervised clustering is performed to create classifier automatically. You can also create classifier manually.

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

## Intermediate result validation
```
python script/HDoG_intermediate.py param_param_example_mergebrain.json CB1_on_850-900_148804_249860_1000-1250_750-1000
```

Second argument of the script defines which part of images to be processed. It is in the following format :

`REGIONNAME_FWorRV_ZLOCALstart-ZLOCALend_YNAME_XNAME_YLOCALstart-YLOCALend_XLOCALstart_XLOCALend`

 * `REGIONNAME` is a label for human
 * `FWorRV` should be either `FW`, `off`, `RV` or `on`
 * `YNAME`,`XNAME` specify which stack to be used.
 * `ZLOCALstart`,`ZLOCALend` specify which images in the stack to be used. 0 corresponds to the first image in the stack.
 * `YLOCALstart`,`YLOCAL_end`,`XLOCALstart`,`XLOCALend` specify which area in images to be used. (0,0) corresponds to the top left pixel.
