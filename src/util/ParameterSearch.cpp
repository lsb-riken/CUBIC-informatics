#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <chrono>
#include "stdio.h"
#include "../3D/NormalizeFilter.cuh"
#include "../3D/DoG3DFilter.cuh"
#include "../3D/Hessian3DFilter_element.cuh"
#include "../3D/CCL3D.cuh"
#include "../3D/RegionalFeatures.cuh"
#include "../3D/utils.h"
#include <json.hpp>
#include <iomanip>

using json = nlohmann::json;

Parameters p;

namespace param {
  struct ParamSearchResult {
    param::SigmaDoG sigma_dog;
    param::RadiusNorm radius_norm;
    float mean;
    float stdev;
  };
  void to_json(json& j, const ParamSearchResult& p) {
    j = json{{"sigma_dog", p.sigma_dog},
             {"radius_norm", p.radius_norm},
             {"mean", p.mean},
             {"std", p.stdev}};
  }
};

// host memory
unsigned short *h_img = NULL;
float *h_avenorm_region = NULL;
int h_num_regions;

// device memory
unsigned short *d_img = NULL;
float *d_hessian = NULL;
char *d_hessian_pd = NULL;
int *d_labels = NULL;
int *d_labels_tmp = NULL;
float *d_hessian_tmp = NULL;
int *d_labels_region = NULL;
int* d_num_regions = NULL;
void *d_cub_tmp = NULL;

// recycled device memory
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_norm = NULL;
float *d_dog = NULL;
float *d_norm_tmp = NULL;
unsigned short *d_size_region = NULL;
float *d_avenorm_region = NULL;
unsigned short *d_erosion_tmp1 = NULL;
unsigned short *d_erosion_tmp2 = NULL;
float *d_erosion_l = NULL;
float *d_dilation_l = NULL;


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
  checkCudaErrors(cudaHostAlloc(&h_avenorm_region, p.image_size*sizeof(float), 0));

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_norm, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_labels, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_tmp, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_region, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian, (6*p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_pd, (p.image_size*sizeof(char))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_tmp, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_size_region, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_num_regions, sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_cub_tmp, p.cub_tmp_bytes));

  // recycle device memory
  d_erosion_tmp1 = reinterpret_cast<unsigned short*>(d_hessian);
  d_erosion_tmp2 = d_erosion_tmp1 + p.image_size;
  d_erosion_l = d_hessian + p.image_size;
  d_dilation_l = d_erosion_l + p.image_size;
  d_temp1 = d_hessian;
  d_temp2 = d_temp1 + p.image_size;
  d_dog = reinterpret_cast<float*>(d_labels_tmp);
  d_avenorm_region = d_hessian + 4*p.image_size;
  d_norm_tmp = d_hessian_tmp;
}

void finalize() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_norm));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_labels_tmp));
  checkCudaErrors(cudaFree(d_labels_region));
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_size_region));
  checkCudaErrors(cudaFree(d_num_regions));
  checkCudaErrors(cudaFree(d_cub_tmp));
  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_avenorm_region));
}

int main(int argc, char** argv) {

  loadParamFile(argv[1], p);

  initialize(p.devID);

  if (p.n_stack != 1) {
    std::cout << "A single stack should be provided!" << std::endl;
    exit(2);
  }
  if(p.list_stack_length[0] != p.depth){
    std::cout << "depth(=" << p.list_stack_length[0] << ") should be " << p.depth << std::endl;
    exit(2);
  }

  // load images to host
  for(int i=0; i < p.depth; i++) {
    loadImage(p.list_src_path[0][i], &h_img[i*p.image_size2D], p.image_size2D);
  }

  // copy images to device
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img, p.image_size*sizeof(unsigned short), cudaMemcpyHostToDevice));

  std::cout << "Number of parameters to test: " << p.n_search << std::endl;

  std::vector<param::ParamSearchResult> list_result(p.n_search);
  for(int i_search = 0; i_search < p.n_search; i_search++) {
    auto s = p.list_sigma_dog[i_search];
    auto r = p.list_radius_norm[i_search];
    initGaussian3DKernel(s.small_xy, s.small_z,
                         s.large_xy, s.large_z);
    list_result[i_search].radius_norm = r;
    list_result[i_search].sigma_dog = s;

    std::cout << "sigma_xy1: " << s.small_xy << "\t"
              << "sigma_z1: "  << s.small_z << "\t"
              << "sigma_xy2: " << s.large_xy << "\t"
              << "sigma_z2: "  << s.large_z << std::endl;
    std::cout << "radius_large_xy: " << r.large_xy << "\t"
              << "radius_large_z: "  << r.large_z <<  std::endl;

    // launch kernels
    Normalize3DFilter(d_img, d_norm, d_erosion_tmp1, d_erosion_tmp2,
                      d_erosion_l, d_dilation_l,
                      p.min_intensity_truncate, p.width, p.height, p.depth,
                      r.large_xy, r.large_z);

    DoG3DFilter(d_norm, d_temp1, d_temp2, d_dog,
                p.width, p.height, p.depth, p.gamma_n);

    HessianPositiveDefiniteWithElement(d_hessian, d_hessian_pd,
                                       d_dog, d_hessian_tmp,
                                         p.width, p.height, p.depth);
    CCL(d_hessian_pd, d_labels, p.width, p.height, p.depth);

    AverageNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
                      d_labels_region, d_avenorm_region,
                      d_size_region, d_cub_tmp, p.cub_tmp_bytes,
                      d_num_regions, h_num_regions,
                      p.width, p.height, p.depth);

    // download only number of regions
    checkCudaErrors(cudaMemcpyAsync((char *)h_avenorm_region, (char *)d_avenorm_region, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
    //barrier (without this, next depth kernel would overlap!!)
    checkCudaErrors(cudaDeviceSynchronize());

    // compute mean and std for nonzero avenorm
    std::vector<float> log_avenorm;
    for(int i=1; i < h_num_regions; i++) {
      if(h_avenorm_region[i] > 0) {
        log_avenorm.push_back(std::log(h_avenorm_region[i]));
      }
    }
    float mean = std::accumulate(log_avenorm.begin(), log_avenorm.end(), 0.0) / log_avenorm.size();
    float stdev = std::sqrt(std::accumulate(log_avenorm.begin(), log_avenorm.end(), 0.0,
                                            [mean](double sum, float e){
                                              return sum + (e - mean) * (e - mean);
                                            }) / log_avenorm.size());
    std::cout << "mean: " << mean << "\tstd: " << stdev << std::endl;

    list_result[i_search].mean = mean;
    list_result[i_search].stdev = stdev;
  }

  // output results as json
  std::string fname = (argc >= 3) ? argv[2] : "result_parameter_search.json";
  std::cout << "Saving to " << fname << std::endl;
  json j(list_result);
  std::ofstream of(fname);
  of << std::setw(4) << j << std::endl;

  finalize();
  return 0;
}
