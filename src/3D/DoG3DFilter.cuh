#ifndef DOG_3D_FILTER_H
#define DOG_3D_FILTER_H


extern "C" int initGaussian3DKernel
(
 const float sigma_xy1, const float sigma_z1,
 const float sigma_xy2, const float sigma_z2
);

extern "C" void Gaussian3DFilter
(
 float *d_img, float *d_temp, float *d_result,
 const int sigma_num, const int width, const int height, const int depth
);

extern "C" void DoG3DFilter
(
 float *d_img,
 float *d_temp1, float *d_temp2,
 float *d_result,
 const int width, const int height, const int depth,
 float gamma
);


#endif
