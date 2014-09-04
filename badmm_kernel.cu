/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/
#include "badmm_kernel.cuh"


__global__ void vecInit(float* X, unsigned int size, float value)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        X[i] = value;
    }
}

__global__ void xexp( float* X, float* C, float* Y, float* Z, unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned long int i = idx; i < size; i += stride) {
        X[i] = Z[i]*__expf(C[i] - Y[i]);
    }
}

__global__ void zexp( float* Z, float* X, float* Y, unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned long int i = idx; i < size; i += stride) {
        Z[i] = X[i]*__expf(Y[i]);
    }
}

__global__ void rowNorm( float* X, float* v, unsigned int size, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int row;

    for (unsigned long int i = idx; i < size; i += stride) {
        row = (int)i/n;
        X[i] /= v[row];
    }
}


__global__ void colNorm( float* X, float* v, unsigned int size, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int col;

    for (unsigned long int i = idx; i < size; i += stride) {
        col = (int)i%n;
        X[i] /= v[col];
    }
}

__global__ void dual( float* err, float* Y, float* X, float* Z, unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    float temp;

    err[idx] = 0.0;

    for (unsigned int i = idx; i < size; i += stride) {
        temp = X[i] - Z[i];
        Y[i] += temp;
        err[idx] += temp*temp;
    }
//    __syncthreads();
}

__global__ void matsub( float* X, float* Y, unsigned int size)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < size; i += stride) {
        X[i] -= Y[i];
    }
}

__global__ void rowNorm_a( float* X, float* v, float* a, unsigned int size, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int row;

    for (unsigned long int i = idx; i < size; i += stride) {
        row = (int)i/n;
        X[i] /= v[row]*a[row];
    }
}

__global__ void colNorm_b( float* X, float* v, float* b, unsigned int size, unsigned int n)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int col;

    for (unsigned long int i = idx; i < size; i += stride) {
        col = (int)i%n;
        X[i] /= v[col]*b[col];
    }
}

__global__ void reduce(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x];
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    // avoid bank conflict
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}