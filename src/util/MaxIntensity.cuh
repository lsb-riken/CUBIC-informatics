#ifndef MAX_INTENSITY_CUH
#define MAX_INTENSITY_CUH

void MaxIntensity
(
 unsigned short *d_img,
 unsigned short *d_maxintensity,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int width, int height, int depth
 );

#endif