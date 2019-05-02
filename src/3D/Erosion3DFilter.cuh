#ifndef EROSION_3D_FILTER_CUH
#define EROSION_3D_FILTER_CUH


void Erosion3DFilter
(
 unsigned short *d_img, unsigned short *d_temp, unsigned short *d_result,
 int w,int h,int d,
 int radius_xy, int radius_z
);


#endif
