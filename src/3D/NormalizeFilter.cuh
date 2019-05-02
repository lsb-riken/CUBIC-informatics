#ifndef NORMALIZE_FILTER_CUH
#define NORMALIZE_FILTER_CUH


void Normalize3DFilter
(
 unsigned short *d_img, float *d_norm,
 unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
 float *d_erosion_l, float *d_dilation_l, float min_erosion_l,
 const int width, const int height, const int depth,
 const int radius_large_xy, const int radius_large_z
 );

#endif
