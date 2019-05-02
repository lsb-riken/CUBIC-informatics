#ifndef HESSIAN_3D_FILTER_ELEMENT_CUH
#define HESSIAN_3D_FILTER_ELEMENT_CUH

void HessianPositiveDefiniteWithElement
(
 float *d_hessian, char *d_hessian_pd, float *d_src, float *d_temp,
 int w, int h, int d
 );

#endif
