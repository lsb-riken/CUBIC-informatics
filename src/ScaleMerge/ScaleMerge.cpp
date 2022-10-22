#include <iostream>
#include <string>
#include "stdio.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdint>
#include <memory>
#include <iterator>
#include <functional>
#include "tinytiffreader.h"
#include "tinytiffwriter.h"


void loadImage(const std::string img_path, std::vector<uint16_t> &img, const int64_t width, const int64_t height,
               const bool vflip, const bool hflip, const bool rotCCW, const bool rotCW, const int64_t imgformat) {
  int64_t width_input, height_input;

  if(rotCW || rotCCW) {
    width_input = height;
    height_input = width;
  }else {
    width_input = width;
    height_input = height;
  }

  //std::cout << "loadImage(" << img_path << ")" << std::endl;
  std::vector<uint16_t> temp(img.size());
  if (img_path.length() == 0) {
    // if empty, fill with zero
    std::fill(img.begin(), img.end(), 0);
  }else {
    if(imgformat == 1) {
      // binary image format
      FILE *f;
      if ((f = fopen(img_path.c_str(), "rb")) == NULL) {
        std::cout << "Unable to open " << img_path << std::endl;
        exit(2);
      }
      size_t ret = fread(reinterpret_cast<char *>(temp.data()), sizeof(unsigned short), img.size(), f);
      if (ret == 0) {
        std::cout << "Unable to read " << img_path << std::endl;
        exit(2);
      }
      fclose(f);
    } else if(imgformat == 0) {
      // tiff image format
      TinyTIFFReaderFile* f = NULL;
      f = TinyTIFFReader_open(img_path.c_str());
      if (!f) {
        std::cout << "Unable to open" << img_path << std::endl;
        exit(2);
      }
      if (width_input != TinyTIFFReader_getWidth(f) || height_input != TinyTIFFReader_getHeight(f) ||
          TinyTIFFReader_getSampleFormat(f) != TINYTIFFREADER_SAMPLEFORMAT_UINT ||
          TinyTIFFReader_getBitsPerSample(f) != 16
        ) {
        std::cout << "Invalid tiff file!" << std::endl;
        std::cout << "the file should be single page uint 16bit with"
                  << "width: " << width_input << ", height:" << height_input << std::endl;
        exit(2);
      }
      TinyTIFFReader_getSampleData(f, temp.data(), 0);
      TinyTIFFReader_close(f);
    } else {
      std::cout << "Invalid imgformat" << std::endl;
      exit(2);
    }
  }

  std::function<std::pair<int,int>(int,int)> transform_rot;
  if(rotCW) {
    transform_rot = [height_input](int x, int y) {
      return std::make_pair((height_input-1-y), x);
    };
  } else if (rotCCW) {
    transform_rot = [width_input](int x, int y) {
      return std::make_pair(y, (width_input-1-x));
    };
  } else {
    transform_rot = [](int x, int y) {
      return std::make_pair(x, y);
    };
  }

  std::function<int(std::pair<int,int>)> transform_flip;
  if (!vflip && !hflip) {
    transform_flip = [width](std::pair<int,int> xy) {
      return xy.second * width + xy.first;
    };
  } else if (!vflip && hflip) {
    transform_flip = [width](std::pair<int,int> xy) {
      return xy.second * width + (width-1-xy.first);
    };
  } else if(vflip && !hflip) {
    transform_flip = [width,height](std::pair<int,int> xy) {
      return (height-1-xy.second)*width + xy.first;
    };
  } else {
    transform_flip = [width,height](std::pair<int,int> xy) {
      return (height-1-xy.second)*width + (width-1-xy.first);
    };
  }


  for(int y = 0; y < height_input; y++)
    for(int x = 0; x < width_input; x++) {
      auto xy = transform_rot(x,y);
      auto i_transformed = transform_flip(xy);
      img[i_transformed] = temp[y*width_input+x];
    }

  // for debug
  //std:copy(img.begin(), img.begin()+10, std::ostream_iterator<uint16_t>(std::cout, ", "));
  //std::cout << std::endl;
}


void saveImage(std::vector<uint16_t> img, const std::string fname, const int64_t width, const int64_t height) {

  TinyTIFFFile* tiff=TinyTIFFWriter_open(fname.c_str(), 16, width, height);
  if(!tiff) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }

  TinyTIFFWriter_writeImage(tiff, img.data());
  TinyTIFFWriter_close(tiff);
}


void loadParam(const std::string param_path,
               std::vector<std::string> &img_paths,
               int64_t &width, int64_t &height,
               int64_t &size_x, int64_t &size_y, double &downscale_ratio,
               int64_t &overlap_left, int64_t &overlap_right,
               int64_t &overlap_top, int64_t &overlap_bottom,
               int64_t &flip_rot_before, int64_t &flip_rot_after,
               int64_t &imgformat, bool &show_grid) {

  std::cout << "loading " << param_path << " ..." << std::endl;

  std::ifstream param_file(param_path);
  if(!param_file.is_open()) {
    std::cout << "Unable to open file" << std::endl;
    exit(2);
  }

  std::string line;
  std::getline(param_file, line);
  std::istringstream stream_line(line);
  std::string num_str;
  //width
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_w(num_str);
  stream_w >> width;
  //height
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_h(num_str);
  stream_h >> height;
  // size_x
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_x(num_str);
  stream_x >> size_x;
  //size_y
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_y(num_str);
  stream_y >> size_y;
  //downscale_ratio
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_sampl(num_str);
  stream_sampl >> downscale_ratio;
  //overlap_left
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_ol(num_str);
  stream_ol >> overlap_left;
  //overlap_right
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_or(num_str);
  stream_or >> overlap_right;
  //overlap_top
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_ot(num_str);
  stream_ot >> overlap_top;
  //overlap_bottom
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_ob(num_str);
  stream_ob >> overlap_bottom;
  //flip_rot_before
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_flb(num_str);
  stream_flb >> flip_rot_before;
  //flip_rot_before
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_fla(num_str);
  stream_fla >> flip_rot_after;
  //imgformat
  std::getline(stream_line, num_str, ':');
  std::istringstream stream_format(num_str);
  stream_format >> imgformat;
  //show_grid
  std::getline(stream_line, num_str, ':');
  if(num_str[0] == '0') {
    show_grid = false;
  } else {
    show_grid = true;
  }

  //std::cout << "size_x: " << size_x << " size_y: " << size_y << std::endl;
  img_paths = std::vector<std::string>(size_x*size_y, "");

  for(int y = 0; y < size_y; y++) {
    for(int x = 0; x < size_x; x++) {
      std::getline(param_file, line);
      //std::cout << "(" << x << ", " << y << " ) line :" << line << " has length " << line.length() << std::endl;
      img_paths[y*size_x+x] = line;
    }
  }
}

int main(int argc, char** argv) {
  if(argc != 3 && argc != 5) {
    std::cout << "Usage:" << std::endl;
    std::cout << argv[0] << " PARAMETERS_FILE MERGED_FILE [MERGED_MAX_FILE] [MERGED_MIN_FILE]" << std::endl;
    exit(2);
  }

  bool output_minmax = (argc == 5);

  // param
  int64_t width;// = 2160;
  int64_t height;// = 2560;
  double downscale_ratio;// = 0.05;
  int64_t overlap_left;// = 100;
  int64_t overlap_right;// = 100;
  int64_t overlap_top;// = 100;
  int64_t overlap_bottom;// = 100;
  int64_t size_x, size_y;
  int64_t flip_rot_before, flip_rot_after, imgformat;
  bool show_grid;
  std::vector<std::string> img_paths;
  loadParam(std::string(argv[1]), img_paths,
            width, height,
            size_x, size_y,
            downscale_ratio, overlap_left, overlap_right,
            overlap_top, overlap_bottom,
            flip_rot_before, flip_rot_after, imgformat, show_grid);

  // computed params
  int64_t strip_width = width - overlap_left - overlap_right;
  int64_t strip_height  = height - overlap_top - overlap_bottom;
  // max int less than or equal strip_width * downscale_ratio
  int64_t sampled_width = strip_width * downscale_ratio;
  int64_t sampled_height = strip_height * downscale_ratio;
  double actual_downscale_ratio_x = sampled_width / strip_width;
  double actual_downscale_ratio_y = sampled_height / strip_height;
  int64_t kernel_width = strip_width / sampled_width;
  int64_t kernel_height = strip_height / sampled_height;
  int64_t merged_width = sampled_width * size_x;
  int64_t merged_height = sampled_height * size_y;
  bool is_hflip_before = (flip_rot_before & 1);
  bool is_vflip_before = (flip_rot_before & 2) >> 1;
  bool is_rotCCW_before = (flip_rot_before & 4) >> 2;
  bool is_rotCW_before = (flip_rot_before & 8) >> 3;
  bool is_hflip_after = (flip_rot_after & 1);
  bool is_vflip_after = (flip_rot_after & 2) >> 1;

  std::cout << "flip v: " << (is_vflip_before ? "on" : "off") << "\thflip: " << (is_hflip_before ? "on" : "off") << std::endl;
  std::cout << "rot ccw: " << (is_rotCCW_before ? "on" : "off") << "\trot cw: " << (is_rotCW_before ? "on" : "off") << std::endl;
  std::cout << "loaded images : (" << width << " x " << height << ") x (" << size_x << " x " << size_y << ")" << std::endl;
  std::cout << "\t overlap L: " << overlap_left << " R: " << overlap_right << " T: " << overlap_top << " B: " << overlap_bottom << std::endl;
  std::cout << "->stripped image : (" << strip_width << " x " << strip_height << ") x (" << size_x << " x " << size_y << ")" << std::endl;
  std::cout << "\t downscale by " << downscale_ratio << "(actual:" << actual_downscale_ratio_x << "," << actual_downscale_ratio_y << ")" << std::endl;
  std::cout << "\t with kernel size: " << kernel_width << " x " << kernel_height << std::endl;
  std::cout << "->sampled images : (" << sampled_width << " x " << sampled_height << ") x (" << size_x << " x " << size_y << ")" << std::endl;
  std::cout << "\t merge" << std::endl;
  std::cout << "->merged image : (" << merged_width << " x " << merged_height << ")" << std::endl << std::endl;

  std::cout << "show grid mode: ";
  if (show_grid)
    std::cout << "ON" << std::endl;
  else
    std::cout << "OFF" << std::endl;

  if (imgformat == 0)
    std::cout << "loading tiff images..." << std::endl;
  else if(imgformat == 1)
    std::cout << "loading binary images..." << std::endl;
  else {
    std::cout << "invalid image format!" << std::endl;
    exit(2);
  }

  std::vector<uint16_t> merged_img_mean(merged_width*merged_height);
  std::vector<uint16_t> merged_img_max(merged_width*merged_height);
  std::vector<uint16_t> merged_img_min(merged_width*merged_height);
  std::vector<uint16_t> img(width*height);
  for(int sy = 0; sy < size_y; sy++) {
    for(int sx = 0; sx < size_x; sx++) {
      loadImage(img_paths[sy*size_x+sx], img, width, height, is_vflip_before, is_hflip_before,
                is_rotCCW_before, is_rotCW_before, imgformat);

      uint16_t *merged_img_part_mean = &merged_img_mean[(sampled_height*sy * merged_width + sampled_width*sx)];
      uint16_t *merged_img_part_max = &merged_img_max[(sampled_height*sy * merged_width + sampled_width*sx)];
      uint16_t *merged_img_part_min = &merged_img_min[(sampled_height*sy * merged_width + sampled_width*sx)];

      // sampling with average
      // each (kernel_width x kernel_height) block is reduce to a top-left pixel
      for(int yy = 0; yy < sampled_height; yy++) {
        for(int xx = 0; xx < sampled_width; xx++) {
          uint32_t val_sum = 0;
          uint16_t val_max = 0;
          uint16_t val_min = UINT16_MAX;
          for(int y = yy*kernel_height + overlap_top; y < (yy+1)*kernel_height + overlap_top; y++) {
            for(int x = xx*kernel_width + overlap_left; x < (xx+1)*kernel_width + overlap_left; x++) {
              val_sum += img[y * width + x];
              val_max = std::max(val_max, img[y * width + x]);
              val_min = std::min(val_min, img[y * width + x]);
            }
          }
          merged_img_part_mean[yy * merged_width + xx] = val_sum / (kernel_width*kernel_height);
          merged_img_part_max[yy * merged_width + xx] = val_max;
          merged_img_part_min[yy * merged_width + xx] = val_min;
        }
        if(show_grid) {
          merged_img_part_mean[yy*merged_width+0] = 5000;
          merged_img_part_max[yy*merged_width+0] = 5000;
          merged_img_part_min[yy*merged_width+0] = 5000;
          if(yy == 0)
            for(int xx=1; xx < sampled_width; xx++) {
              merged_img_part_mean[yy*merged_width+xx] = 5000;
              merged_img_part_max[yy*merged_width+xx] = 5000;
              merged_img_part_min[yy*merged_width+xx] = 5000;
            }
        }
      }
    }
  }
  std::cout << "saving merged image..." << std::endl;
  std::cout << "flip v: " << (is_vflip_after ? "on" : "off") << "\t flip h: " << (is_hflip_after ? "on" : "off") << std::endl;

  std::vector<uint16_t> merged_img_flip_mean(merged_width*merged_height);
  std::vector<uint16_t> merged_img_flip_max(merged_width*merged_height);
  std::vector<uint16_t> merged_img_flip_min(merged_width*merged_height);
  if(!is_vflip_after && !is_hflip_after) {
    for(int sy = 0; sy < merged_height; sy++) {
      for(int sx = 0; sx < merged_width; sx++) {
        merged_img_flip_mean[sy*merged_width+sx] = merged_img_mean[sy*merged_width+sx];
        merged_img_flip_max[sy*merged_width+sx] = merged_img_max[sy*merged_width+sx];
        merged_img_flip_min[sy*merged_width+sx] = merged_img_min[sy*merged_width+sx];
      }
    }
  } else if (!is_vflip_after && is_hflip_after) {
    for(int sy = 0; sy < merged_height; sy++) {
      for(int sx = 0; sx < merged_width; sx++) {
        merged_img_flip_mean[sy*merged_width+(merged_width-1-sx)] = merged_img_mean[sy*merged_width+sx];
        merged_img_flip_max[sy*merged_width+(merged_width-1-sx)] = merged_img_max[sy*merged_width+sx];
        merged_img_flip_min[sy*merged_width+(merged_width-1-sx)] = merged_img_min[sy*merged_width+sx];
      }
    }
  } else if(is_vflip_after && !is_hflip_after) {
    for(int sy = 0; sy < merged_height; sy++) {
      for(int sx = 0; sx < merged_width; sx++) {
        merged_img_flip_mean[(merged_height-1-sy)*merged_width+sx] = merged_img_mean[sy*merged_width+sx];
        merged_img_flip_max[(merged_height-1-sy)*merged_width+sx] = merged_img_max[sy*merged_width+sx];
        merged_img_flip_min[(merged_height-1-sy)*merged_width+sx] = merged_img_min[sy*merged_width+sx];
      }
    }
  } else {
    for(int sy = 0; sy < merged_height; sy++) {
      for(int sx = 0; sx < merged_width; sx++) {
        merged_img_flip_mean[(merged_height-1-sy)*merged_width+(merged_width-1-sx)] = merged_img_mean[sy*merged_width+sx];
        merged_img_flip_max[(merged_height-1-sy)*merged_width+(merged_width-1-sx)] = merged_img_max[sy*merged_width+sx];
        merged_img_flip_min[(merged_height-1-sy)*merged_width+(merged_width-1-sx)] = merged_img_min[sy*merged_width+sx];
      }
    }
  }

  saveImage(merged_img_flip_mean, std::string(argv[2]), merged_width, merged_height);
  if(output_minmax) {
    saveImage(merged_img_flip_max, std::string(argv[3]), merged_width, merged_height);
    saveImage(merged_img_flip_min, std::string(argv[4]), merged_width, merged_height);
  }

  return 0;
}
