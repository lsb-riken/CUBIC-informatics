#ifndef HDOG_UTILS_H
#define HDOG_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "stdio.h"
#include <json.hpp>

using json = nlohmann::json;

namespace param {
  struct SigmaDoG {
    float small_xy;
    float small_z;
    float large_xy;
    float large_z;
  };
  struct RadiusNorm {
    int large_xy;
    int large_z;
  };
  void to_json(json& j, const SigmaDoG& p) {
    j = json{{"xy_small", p.small_xy},
             {"z_small", p.small_z},
             {"xy_large", p.large_xy},
             {"z_large", p.large_z}};
  }
  void from_json(const json& j, SigmaDoG& p) {
    j.at("xy_small").get_to(p.small_xy);
    j.at("z_small").get_to(p.small_z);
    j.at("xy_large").get_to(p.large_xy);
    j.at("z_large").get_to(p.large_z);
  }
  void to_json(json& j, const RadiusNorm& p) {
    j = json{{"xy_large", p.large_xy},
             {"z_large", p.large_z}};
  }
  void from_json(const json& j, RadiusNorm& p) {
    j.at("xy_large").get_to(p.large_xy);
    j.at("z_large").get_to(p.large_z);
  }
}

/*  Parameters */
struct Parameters {
  // GPU Device ID
  int devID;
  bool verbose = false;

  // size [pixel]
  int width = 2560;
  int height = 2160;
  int depth = 32;
  int image_size; // = width * height * depth
  int image_size2D; // = width*height
  // cub_tmp_size_factor = 8.001
  // assertion fail if this is smaller than required
  float cub_tmp_size_factor = 8.001;
  size_t cub_tmp_bytes; // = image_size * cub_tmp_size_factor

  // margin [pixel]
  int left_margin = 100;
  int right_margin = 100;
  int top_margin = 100;
  int bottom_margin = 100;
  int depth_margin; // automatically determined
  int extra_depth_margin = 3;

  // scale [um/pixel] nagative value to flip the direction
  float scale_x = 1.0;
  float scale_y = 1.0;
  float scale_z = 1.0;

  // algorithm parameters
  float gamma_n = 1.0;
  param::SigmaDoG sigma_dog;
  param::RadiusNorm radius_norm;
  unsigned short min_intensity_skip = 1000;
  unsigned short min_intensity_truncate = 1000;
  bool is_ave_mode = false;

  // threshold to reduce result filesize
  unsigned short min_size = 0;

  // stacks
  int n_stack;
  std::vector<std::vector<std::string> > list_src_path;
  std::vector<int> list_stack_length;
  std::vector<std::string> list_dst_path;

  // parameter search
  int n_search;
  std::vector<param::RadiusNorm> list_radius_norm;
  std::vector<param::SigmaDoG> list_sigma_dog;
};

void loadParamFile(const std::string fname, Parameters &p) {
  std::ifstream param_file(fname);
  if(!param_file.is_open()) {
    std::cout << "Unable to open file." << std::endl;
    exit(2);
  }
  json j;
  param_file >> j;
  j.at("devID").get_to(p.devID);
  try {
    j.at("verbose").get_to(p.verbose);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }

  json j_image = j.at("input_image_info");
  j_image.at("width").get_to(p.width);
  j_image.at("height").get_to(p.height);
  j_image.at("left_margin").get_to(p.left_margin);
  j_image.at("right_margin").get_to(p.right_margin);
  j_image.at("top_margin").get_to(p.top_margin);
  j_image.at("bottom_margin").get_to(p.bottom_margin);

  json j_coord = j.at("coordinate_info");
  j_coord.at("scale_x").get_to(p.scale_x);
  j_coord.at("scale_y").get_to(p.scale_y);
  j_coord.at("scale_z").get_to(p.scale_z);

  json j_param = j.at("HDoG_param");
  j_param.at("depth").get_to(p.depth);
  p.image_size = p.width * p.height * p.depth;
  p.image_size2D = p.width * p.height;
  j_param.at("extra_depth_margin").get_to(p.extra_depth_margin);
  try {
    j.at("cub_tmp_size_factor").get_to(p.cub_tmp_size_factor);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }
  p.cub_tmp_bytes = p.image_size * p.cub_tmp_size_factor;

  try {
    j_param.at("min_intensity_skip").get_to(p.min_intensity_skip);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }
  std::cout << "min_intensity_skip:" << p.min_intensity_skip << std::endl;

  j_param.at("radius_norm").get_to(p.radius_norm);
  j_param.at("min_intensity").get_to(p.min_intensity_truncate);
  std::cout << "min_intensity:" << p.min_intensity_truncate << std::endl;

  j_param.at("gamma_n").get_to(p.gamma_n);
  j_param.at("sigma_dog").get_to(p.sigma_dog);

  j_param.at("min_size").get_to(p.min_size);
  std::cout << "min_size:" << p.min_size << std::endl;

  try {
    j.at("is_ave_mode").get_to(p.is_ave_mode);
  } catch(nlohmann::detail::out_of_range) {
    // nothing
  }

  json j_stacks = j.at("stacks");
  p.n_stack = j_stacks.size();
  for (json::iterator j_st = j_stacks.begin(); j_st != j_stacks.end(); ++j_st) {
    json j_src = j_st->at("src_paths");
    p.list_stack_length.push_back(j_src.size());
    p.list_src_path.emplace_back();
    j_src.get_to(p.list_src_path.back());
    p.list_dst_path.push_back(j_st->at("dst_path").get<std::string>());
  }

  try {
    json j_search = j.at("parameter_search");
    p.n_search = j_search.size();
    for (json::iterator j_s = j_search.begin(); j_s != j_search.end(); ++j_s) {
      // radius norm
      try {
        json j_search_norm = j_s->at("radius_norm");
        p.list_radius_norm.push_back(j_search_norm.get<param::RadiusNorm>());
      } catch(nlohmann::detail::out_of_range) {
        p.list_radius_norm.push_back(p.radius_norm);
      }
      // sigma dog
      try {
        json j_search_dog = j_s->at("sigma_dog");
        p.list_sigma_dog.push_back(j_search_dog.get<param::SigmaDoG>());
      } catch(nlohmann::detail::out_of_range) {
        p.list_sigma_dog.push_back(p.sigma_dog);
      }
    }
  } catch(nlohmann::detail::out_of_range) {
    p.n_search = 0;
  }

  return;
}

/* Binary Images */
void loadImage(const std::string fname, const unsigned short *h_img, const int image_size2D) {
  //std::cout << "loadImage(" << fname << ")" << std::endl;
  FILE *f;
  if ((f = fopen(fname.c_str(), "rb")) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }
  fread((char *)h_img, sizeof(unsigned short), image_size2D, f);
  fclose(f);
}

template<typename T>
void saveImage(const T* img_result, const std::string fname, const long long size, const std::string mode="wb") {
  //std::cout << "saveImage(" << fname << ")" << std::endl;
  FILE *f;
  if ((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }
  fwrite((char *)img_result, sizeof(T), size, f);
  fclose(f);
}

/*  Feature Vector */
struct FeatureVector{
  float centroid_x;
  float centroid_y;
  float centroid_z;
  float structureness;
  float blobness;
  float max_normalized;
  unsigned short size;
  unsigned short padding;
};

void loadFeatureVector(const std::string fname, const Parameters &p,
                       int z0,
                       float offset_x, float offset_y, float offset_z) {
}

void saveFeatureVector
(
 float *h_maxnorm_region,
 unsigned short *h_size_region,
 float *h_eigen_region, float *h_grid_region,
 const std::string fname, int num_regions,
 int depth, int z0, const Parameters &p,
 const std::string mode="wb", float min_maxnorm=0
 ) {
  //std::cout << "saveFeatureVector(" << fname << "," << num_regions << "," << depth << "," << z0 << "," << mode << ")" << std::endl;

  FILE *f;
  if((f = fopen(fname.c_str(), mode.c_str())) == NULL) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }

  int count = 0;
  for(int i = 1; i < num_regions; i++) { // skip background(i=0)
    float centroid_x = h_grid_region[i];
    float centroid_y = h_grid_region[p.image_size+i];
    float centroid_z = h_grid_region[2*p.image_size+i];
    //std::cout << centroid_x << "," << centroid_y << "," << centroid_z << std::endl;
    // margins
    if (centroid_z < p.depth_margin || centroid_z >= depth-p.depth_margin) continue;
    if (centroid_x < p.left_margin || centroid_x >= p.width-p.right_margin) continue;
    if (centroid_y < p.top_margin || centroid_y >= p.height-p.bottom_margin) continue;
    // threshold
    if (h_size_region[i] < p.min_size || h_maxnorm_region[i] <= min_maxnorm) continue;

    // save binary FeatureVector struct
    FeatureVector fv = {
      centroid_x,
      centroid_y,
      centroid_z+z0,
      h_eigen_region[i],
      h_eigen_region[p.image_size+i],
      h_maxnorm_region[i],
      h_size_region[i],
      0
    };
    count++;
    fwrite(&fv, sizeof(FeatureVector), 1, f);
  }
  std::cout << "saved count:" << count << "*" << sizeof(FeatureVector) << "=" << count*sizeof(FeatureVector) << std::endl;
  fclose(f);
}


#endif
