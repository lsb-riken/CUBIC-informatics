# CUBIC-informatics

Relevant Paper:
> Matsumoto, K., Mitani, T.T., Horiguchi, S.A. et al. Advanced CUBIC tissue clearing for whole-organ cell profiling. Nat Protoc 14, 3506–3537 (2019) [doi:10.1038/s41596-019-0240-9](https://doi.org/10.1038/s41596-019-0240-9)

Tutorial for cell/nucleus detection: [see this notebook](https://github.com/lsb-riken/CUBIC-informatics/blob/master/demo.ipynb)


## UPDATE: Docker
Since the original code was dependent on CUDA 9.0, which is too old now, we have updated the programs to work on CUDA 11.6.

Also, we added `Dockerfile` and `docker-compose.yml`. 
Once you install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker), you can easily get the environment already set up with python and compiled CUDA programs. You do not need to manually install the CUDA toolkit and build the source code on the host.
Please refer to [this installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for NVIDIA Container Toolkit and [this guide](https://docs.docker.com/get-started/) for Docker engine.

We recommend using [docker-compose](https://docs.docker.com/compose/) to simplify the handling of Docker containers. We explain below how to use the Docker container with docker-compose.

Once you downloaded this repository by following _Downloading source codes and reference data_ section below,
Run the following command in the `CUBIC-informatics` directory to build the container.

```
    docker compose build
```

Then, skip _Building C++/CUDA programs_ section.
In _Cell candidate detection_ section, run the following commands (add `docker compose run dev` in front). The computations will start in the container

```
    docker compose run dev python script/HDoG_gpu.py param/param_example_HDoG_FW.json
    docker compose run dev python script/HDoG_gpu.py param/param_example_HDoG_RV.json
```

Since other steps (_Candidate verification_, _Registration_, etc.) do not require the CUDA environment, you can run the python scripts on the host or in the container.



## Getting Started

### Hardware Prerequisites
 * One or more NVIDIA GPUs (tested on [compute capability](https://developer.nvidia.com/cuda-gpus) 6.1 and 5.2)

Note that the GPU memory size is critical for performance.
If your GPUs have less than 11GB memory, you need to adjust not only the parameters in parameter file but also the costants defined in the source code. c.f. [#2](https://github.com/lsb-riken/CUBIC-informatics/issues/2)

### Software Prerequisites
Tested on Cent OS 7.5 with the following versions of software.

 * Cmake 2.8.12
 * GNU Compiler Collection 4.8.5
 * Python 3.6.4
 * Julia 0.6.3 or 1.20
 * [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 9.0
 * [ANTs](https://github.com/ANTsX/ANTs/releases) 2.1.0

### Source Image Prerequisites

1. Source images should be in binary format. Batch image conversion from TIFF to binary can be performed with `script/tiff2bin.py`.

2. Source image files are assumed to be named like:

  `/path/to/src/YNAME/YNAME_XNAME/ZNAME.bin`

 * `XNAME`,`YNAME` is the number that specifies the origin of the stack
 * `ZNAME` is the slice number

Conversion rule from these names to the physical position should be provided in the _HDoG_ parameter files.

### Downloading source codes and reference data
```
git clone https://github.com/lsb-riken/CUBIC-informatics
cd CUBIC-informatics
```

Download CUBIC-Atlas reference data from [here](http://cubic-atlas.riken.jp/).

### Building C++/CUDA programs
Some programs use helper functions in CUDA Toolkit. Create a symbolic link in the top directory(`CUBIC-informatics/`). The path of CUDA Toolkit depends on your system setup.

```
ln -s /usr/local/cuda/samples/common common
```

Build programs under `build` directory. Please change `CUDA_NVCC_FLAGS` in `CMakeLists.txt` to match the [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU devices.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. ..
make
make install
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

3. Preparing _Classify_ parameter file

    example: `param/param_example_classify.json`

4. Preparing _HDoG_ and _Merge_ parameter file as well for other channel images

5. Preparing _MultiChannel_ parameter file

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
Cell candidate detection is performed on GPUs for the forward and reverse sides.

```
python script/MergeBrain.py cells param/param_example_mergebrain.json
```
Candidate detection results on both sides are merged.

### Candidate verification
```
python script/HDoG_classify.py param/param_example_classify.json
```
Create a classifier with manual decision boundary and plot cell candidates in feature space. By setting `use_manual_boundary` in the parameter file to be `false`, you can also perform unsupervised clustering and create a classifier automatically.

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

The second argument of the script defines which part of the images to be processed. It is in the following format :

`REGIONNAME_FWorRV_ZLOCALstart-ZLOCALend_YNAME_XNAME_YLOCALstart-YLOCALend_XLOCALstart_XLOCALend`

 * `REGIONNAME` is a label for human
 * `FWorRV` should be either `FW`, `off`, `RV` or `on`
 * `YNAME`,`XNAME` specify which stack to be used.
 * `ZLOCALstart`,`ZLOCALend` specify which images in the stack to be used. 0 corresponds to the first image in the stack.
 * `YLOCALstart`,`YLOCAL_end`,`XLOCALstart`,`XLOCALend` specify which area in images to be used. (0,0) corresponds to the top left pixel.

## Advanced Usage
If you want to check how the algorithm works, there is a program to test each step in the algorithm. As a python implementation, we have `script/HDoG_cpu.py`.

Also, we have CUDA programs for each step. To build these programs, run `make DoG_test`, for example, and the binary is created under `build/test/` directory.
