/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/

#ifndef _badmm_kernel_h
#define _badmm_kernel_h

#include <cuda.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#define PRECESION   1.0e-10

struct xexp_func
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // X[i] = Z[i] * exp(C[i] - Y[i])
        // 1 = 2 * exp(0-3)
        thrust::get<1>(t) = thrust::get<2>(t) * ( __expf( thrust::get<0>(t) - thrust::get<3>(t) ));
    }
};

struct zexp_func
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // Z[i] = X[i] * exp(Y[i])
        // 2 = 0 * exp(1)
        thrust::get<2>(t) = thrust::get<0>(t) * __expf( thrust::get<1>(t) );
    }
};

struct zdiffsq{
template <typename Tuple>
  __host__ __device__ float operator()(Tuple a)
  {
    float result = thrust::get<0>(a) - thrust::get<1>(a);
    return result*result;
  }
};

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

struct diffsq{
  __host__ __device__ float operator()(float a, float b)
  {
    return (b-a)*(b-a);
  }
};

__global__ void vecInit(float* X, unsigned int size, float value);
//__global__ void objval( float* C, float* X, unsigned int size);
__global__ void xexp( float* X, float* C, float* Y, float* Z, unsigned int size);
__global__ void zexp( float* Z, float* X, float* Y, unsigned int size);
__global__ void rowNorm (float* X, float* v, unsigned int size, unsigned int n);
__global__ void rowNorm_a( float* X, float* v, float* a, unsigned int size, unsigned int n);
__global__ void colNorm (float* Z, float* v, unsigned int size, unsigned int n);
__global__ void colNorm_b( float* X, float* v, float* b, unsigned int size, unsigned int n);
__global__ void dual(float* err, float* Y, float* X, float* Z, unsigned int size);
__global__ void reduce(float *g_idata, float *g_odata, unsigned int size);
__global__ void matrowSum(float* rowsum, float* X, unsigned int m, unsigned int n);
__global__ void matsub( float* X, float* Y, unsigned int size);

#endif