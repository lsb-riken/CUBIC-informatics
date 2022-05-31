#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <string>
#include <chrono>
#include "DoG3DFilter.cuh"
#include "Hessian3DFilter_element.cuh"
#include "CCL3D.cuh"
#include "Erosion3DFilter.cuh"
#include "RegionalFeatures.cuh"
#include "Eigenvalues.cuh"
#include "NormalizeFilter.cuh"
#include "utils.h"

Parameters p;

// host memory
unsigned short *h_img = NULL;
int *h_labels = NULL;
int *h_labels_region = NULL;
unsigned short *h_size_region = NULL;
float *h_maxnorm_region = NULL;
float *h_eigen_region = NULL;
float *h_grid_region = NULL;
int h_num_regions;

// device memory
unsigned short *d_img = NULL;
float *d_hessian = NULL;
char *d_hessian_pd = NULL;
int *d_labels = NULL;
int *d_labels_tmp = NULL;
float *d_hessian_tmp = NULL;
int *d_labels_region = NULL;
unsigned short *d_size_region = NULL;
float *d_eigen_region = NULL;
int* d_num_regions = NULL;
void *d_cub_tmp = NULL;

// recycled device memory
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_norm = NULL;
float *d_dog = NULL;
float *d_hessian_region = NULL;
float *d_grid = NULL;
float *d_grid_region = NULL;
float *d_grid_tmp = NULL;
float *d_norm_tmp = NULL;
float *d_maxnorm_region = NULL;
unsigned short *d_img_tmp = NULL;
unsigned short *d_erosion_tmp1 = NULL;
unsigned short *d_erosion_tmp2 = NULL;
float *d_erosion_l = NULL;
float *d_dilation_l = NULL;

void initializeGPU() {
  // cuda device init
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  cudaSetDevice(p.devID);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, p.devID));
  //checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  cudaFree(0); // to measure cudaMalloc correctly

  // allocating page-locked host memory
  // h_img: WriteCombined is fast for host->device only
  h_img = (unsigned short *)malloc(p.image_size*sizeof(unsigned short));
  h_labels = (int *)malloc(p.image_size*sizeof(int));
  h_labels_region = (int *)malloc(p.image_size*sizeof(int));
  h_maxnorm_region = (float *)malloc(p.image_size*sizeof(float));
  h_size_region = (unsigned short *)malloc(p.image_size*sizeof(unsigned short));
  h_eigen_region = (float *)malloc(2*p.image_size*sizeof(float));
  h_grid_region = (float *)malloc(3*p.image_size*sizeof(float));

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
  checkCudaErrors(cudaMalloc((void **)&d_eigen_region, (2*p.image_size*sizeof(float))));
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
  d_hessian_region = d_hessian;
  d_grid = d_hessian;
  d_grid_region = d_hessian + p.image_size;
  d_maxnorm_region = d_hessian + 4*p.image_size;
  d_grid_tmp = d_hessian_tmp;
  d_norm_tmp = d_hessian_tmp;
}

void finalizeGPU() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_norm));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_labels_tmp));
  checkCudaErrors(cudaFree(d_labels_region));
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_size_region));
  checkCudaErrors(cudaFree(d_eigen_region));
  checkCudaErrors(cudaFree(d_num_regions));
  checkCudaErrors(cudaFree(d_cub_tmp));
  free(h_img);
  free(h_labels);
  free(h_labels_region);
  free(h_size_region);
  free(h_eigen_region);
  free(h_grid_region);
}


void processSubstack(const int i_stack, const int z0) {
  std::chrono::system_clock::time_point  start, end;
  double elapsed;

  int depth = std::min(p.depth, p.list_stack_length[i_stack]-z0);
  int image_size = p.image_size2D * depth;
  bool is_last = depth != p.depth;
  std::string* list_src_path = &(p.list_src_path[i_stack])[z0];

  std::cout << "----------" << std::endl;
  std::cout << "processSubstack(z0=" << z0 << ", depth=" << depth << ")" << std::endl;
  std::cout << "\tfirst_src_path: " << list_src_path[0] << std::endl;
  std::cout << "\tlast_src_path: " << list_src_path[depth-1] << std::endl;

  if(depth == 0) {
    std::cout << "No img provided for processSubstack()!" << std::endl;
    return;
  }

  start = std::chrono::system_clock::now();
  // load images to host memory
  for(int i=0; i < depth; i++) {
    loadImage(list_src_path[i], &h_img[i*p.image_size2D], p.image_size2D);
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "loadImage took " << elapsed << "msec" << std::endl;

  start = std::chrono::system_clock::now();
  // copy images to device
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img,
                                  image_size*sizeof(unsigned short), cudaMemcpyHostToDevice));
  // zero-fill if substack is not full
  if(depth < p.depth)
    checkCudaErrors(cudaMemsetAsync((unsigned short *)&d_img[image_size], 0,
                                    (p.image_size-image_size)*sizeof(unsigned short)));

  // launch kernels
  Normalize3DFilter(d_img, d_norm, d_erosion_tmp1, d_erosion_tmp2,
                    d_erosion_l, d_dilation_l,
                    p.min_intensity_truncate, p.width, p.height, p.depth,
                    p.radius_norm.large_xy, p.radius_norm.large_z);
  DoG3DFilter(d_norm, d_temp1, d_temp2, d_dog,
              p.width, p.height, p.depth, p.gamma_n);
  HessianPositiveDefiniteWithElement(d_hessian, d_hessian_pd,
                                     d_dog, d_hessian_tmp,
                                     p.width, p.height, p.depth);
  CCL(d_hessian_pd, d_labels, p.width, p.height, p.depth);

  HessianFeatures(d_labels, d_hessian,
                  d_labels_tmp, d_hessian_tmp,
                  d_labels_region, d_hessian_region,
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, p.width, p.height, p.depth);
  // get num_regions in host
  checkCudaErrors(cudaMemcpy(&h_num_regions, d_num_regions, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "num_regions: " << h_num_regions << std::endl;

  Eigenvalues(d_hessian_region, d_eigen_region, h_num_regions, p.image_size);

  if (!p.is_ave_mode) {
    MaxNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
                  d_labels_region, d_maxnorm_region,
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, p.width, p.height, p.depth);
  } else {
    SumNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
                  d_labels_region, d_maxnorm_region,
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, p.width, p.height, p.depth);
  }

  RegionalSizeAndCentroid(d_labels, d_grid,
                          d_labels_tmp, d_grid_tmp,
                          d_labels_region, d_size_region, d_grid_region,
                          d_cub_tmp, p.cub_tmp_bytes,
                          d_num_regions, h_num_regions,
                          p.width, p.height, p.depth);

  // download to check the result
  //checkCudaErrors(cudaMemcpyAsync((char *)h_labels, (char *)d_labels, image_size*sizeof(int), cudaMemcpyDeviceToHost));
  // download only number of regions
  checkCudaErrors(cudaMemcpyAsync((char *)h_size_region, (char *)d_size_region, h_num_regions*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_maxnorm_region, (char *)d_maxnorm_region, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 3; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_grid_region+i*p.image_size), (char *)(d_grid_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }
  for(int i = 0; i < 2; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_eigen_region+i*p.image_size), (char *)(d_eigen_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }

  //barrier (without this, next substack kernel call would overlap!!)
  checkCudaErrors(cudaDeviceSynchronize());
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "GPU memcpy & kernel execution took " << elapsed << "msec" << std::endl;


  start = std::chrono::system_clock::now();
  saveFeatureVector(h_maxnorm_region,
                    h_size_region,
                    h_eigen_region, h_grid_region,
                    p.list_dst_path[i_stack],
                    h_num_regions, depth, z0, p, "ab");

  if(!is_last) {
    // middle of the stack
    std::cout << "[save] middle of the stack : "
              << z0 + p.depth_margin << " - " << z0 + depth - p.depth_margin - 1
              << " (" << depth - 2*p.depth_margin << ")" << std::endl;
    //saveImage(h_labels+p.depth_margin*p.image_size2D, dst_path, p.image_size2D*(depth-2*p.depth_margin), "ab");
  } else {
    // end of the stack
    std::cout << "[save] end of the stack : "
              << z0 + p.depth_margin << " - " << z0 + depth - 1
              << " (" << depth - p.depth_margin << ")" << std::endl;
    //saveImage(h_labels+p.depth_margin*p.image_size2D, dst_path, p.image_size2D*(depth-p.depth_margin), "ab");
  }
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "saveImage took " << elapsed << "msec" << std::endl;

}


int main(int argc, char** argv) {
  std::chrono::system_clock::time_point  start, end;
  double elapsed;

  if( argc != 2 ) {
    std::cout << "Usage:" << std::endl;
    std::cout << argv[0] << " PARAM_FILE" << std::endl;
    exit(2);
  }

  loadParamFile(argv[1], p);
  std::cout << "number of stacks: " << p.n_stack << std::endl;

  start = std::chrono::system_clock::now();
  initializeGPU(); // GPU device ID
  end = std::chrono::system_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  std::cout << "Initialization took " << elapsed << "msec" << std::endl;

  // create DoG kernel, and set depth_margin
  int radius_dog = initGaussian3DKernel(p.sigma_dog.small_xy,
                                        p.sigma_dog.small_z,
                                        p.sigma_dog.large_xy,
                                        p.sigma_dog.large_z);
  p.depth_margin = radius_dog + p.extra_depth_margin;

  std::cout << "sigma_xy1: " << p.sigma_dog.small_xy << "\t"
            << "sigma_z1: "  << p.sigma_dog.small_z << "\t"
            << "sigma_xy2: " << p.sigma_dog.large_xy << "\t"
            << "sigma_z2: "  << p.sigma_dog.large_z << std::endl;
  std::cout << "radius_large_xy: " << p.radius_norm.large_xy << "\t"
            << "radius_large_z: "  << p.radius_norm.large_z <<  std::endl;
  std::cout << "depth_margin: " << p.depth_margin << std::endl;

  for(int i_stack = 0; i_stack < p.n_stack; i_stack++) {
    if (remove(p.list_dst_path[i_stack].c_str()) == 0) {
      // because result is written in append mode, the file is removed beforehand
      std::cout << "removed previous result " << i_stack << std::endl;
    } else {
      std::cout << "newly create result " << i_stack << std::endl;
    }
    std::cout << "length: " << p.list_stack_length[i_stack] << std::endl;

    // one stack(`length` images) is partitioned into many substacks.
    // substacks(`depth` images) have overlap with neighboring substacks.
    // for each substack, images [0,depth-1] are used, but regions whose
    // centroid_z is in [0,margin-1] or [depth-margin,depth-1] is discarded.
    // Thus, effective z range is [margin,depth-margin-1].
    // For substacks at the end of stack, end margin is not discarded.
    for(int z0 = 0;
        z0 < p.list_stack_length[i_stack] - p.depth_margin;
        z0 += (p.depth-2*p.depth_margin)) {
      // index z0 indicate the top of substack in stack-wide coordinate
      processSubstack(i_stack, z0);
    }
  }

  finalizeGPU();
  return 0;
}
