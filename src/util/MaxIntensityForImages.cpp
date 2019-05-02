#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <json.hpp>
#include "../3D/utils.h"
#include "./MaxIntensity.cuh"


using json = nlohmann::json;

Parameters p;

// host memory
unsigned short *h_img = NULL;
unsigned short *h_maxintensity = NULL;

// device memory
unsigned short *d_img = NULL;
unsigned short *d_maxintensity = NULL;
void *d_cub_tmp = NULL;


void initialize(const int devID) {
  // cuda device init
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  cudaSetDevice(devID);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  //checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  cudaFree(0); // to measure cudaMalloc correctly


  // allocating page-locked host memory
  // h_img: WriteCombined is fast for host->device only
  checkCudaErrors(cudaHostAlloc(&h_img, p.image_size*sizeof(unsigned short), cudaHostAllocWriteCombined));
  checkCudaErrors(cudaHostAlloc(&h_maxintensity, p.depth*sizeof(unsigned short), 0));

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_maxintensity, (p.depth*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_cub_tmp, p.cub_tmp_bytes));

}

void finalize() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_maxintensity));
  checkCudaErrors(cudaFree(d_cub_tmp));
  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_maxintensity));
}

int main(int argc, char** argv) {

  loadParamFile(argv[1], p);
  std::cout << "number of stacks: " << p.n_stack << std::endl;

  initialize(p.devID);

  std::vector<std::vector<unsigned short> > list_result(p.n_stack);
  for(int i_stack = 0; i_stack < p.n_stack; i_stack++) {
    list_result[i_stack].reserve(p.list_stack_length[i_stack]);
    for(int z0 = 0; z0 < p.list_stack_length[i_stack]; z0 += p.depth) {
      // index z0 indicate the top of substack in stack-wide coordinate
      // load images to host
      for(int i=0; i < p.depth; i++) {
        loadImage(p.list_src_path[i_stack][z0+i], &h_img[i*p.image_size2D], p.image_size2D);
      }
      // copy images to device
      checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img, p.image_size*sizeof(unsigned short), cudaMemcpyHostToDevice));
      // launch kernels
      MaxIntensity(d_img, d_maxintensity,
                   d_cub_tmp, p.cub_tmp_bytes,
                   p.width, p.height, p.depth);

      // download
      checkCudaErrors(cudaMemcpyAsync((char *)h_maxintensity,
                                      (char *)d_maxintensity,
                                      p.depth*sizeof(unsigned short),
                                      cudaMemcpyDeviceToHost));
      for(int i=0; i < p.depth; i++) {
        list_result[i_stack].push_back(h_maxintensity[i]);
      }
      //barrier (without this, next depth kernel would overlap!!)
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // output results as binary
  std::string fname = (argc >= 3) ? argv[2] : "result_maxintensity.bin";
  std::cout << "Saving to " << fname << std::endl;
  std::ofstream ofs(fname, std::ios::binary);
  if (!ofs) {
    std::cout << "Could not open the file." << std::endl;
    return 0;
  }
  for(int i_stack = 0; i_stack < p.n_stack; i_stack++) {
    ofs.write(reinterpret_cast<char*>(&list_result[i_stack][0]),
              sizeof(unsigned short) * p.list_stack_length[i_stack]);
  }

  finalize();
  return 0;
}
