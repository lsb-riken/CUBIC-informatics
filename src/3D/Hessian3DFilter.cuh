#ifndef HESSIAN_3D_FILTER_CUH
#define HESSIAN_3D_FILTER_CUH

extern "C" void HessianPositiveDefinite(
    char *d_hessian_pd,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
);

#endif