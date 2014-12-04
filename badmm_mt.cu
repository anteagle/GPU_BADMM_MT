/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/

#include <stdio.h>
#include "badmm_kernel.cuh"
#include "cublas.h"
#include <math.h>

//#define MAX_GRID_SIZE 65535
//#define MAX_BLOCK_SIZE 1024

typedef struct GPUInfo
{
    unsigned int MAX_GRID_SIZE;
    unsigned int MAX_BLOCK_SIZE;
}GPUInfo;

typedef struct ADMM_para
{
    float rho;          // penalty parameter

    float* iter_obj;
    float* iter_time;
    float* iter_err;
    unsigned int MAX_ITER;          // MAX_ITER
    float tol;
}ADMM_para;

typedef struct BADMM_massTrans
{
    int m;
    int n;
    
    float* C;                       // row major order
    float* a;
    float* b;

    int print_step;

    bool SAVEFILE;
}BADMM_massTrans;

void matInit(float* &X, unsigned int size, float value);

/*********************************************
Bregman ADMM for mass transportation problem
All matrices are in row major order
**********************************************/
void gpuBADMM_MT( BADMM_massTrans* &badmm_mt, ADMM_para* &badmmpara, GPUInfo* gpu_info)
{
    float *X, *Z, *Y;                   // host

    float *d_C, *d_X, *d_Z, *d_Y;       // device
    float *d_a, *d_b;

    float *d_Xold, *d_Yerr;             // for stopping condition
    float Xerr, Yerr;
    
    float *d_rowSum, *d_colSum;         // for row and column normalization
    float *col_ones, *row_ones;

    unsigned int m,n;
    m = badmm_mt->m;
    n = badmm_mt->n;
    
    unsigned long int size = m*n;
    
    // initialize host matrix
    X = new float[size];
    Z = new float[size];
    Y = new float[size];
    
    matInit(X,size,0.0);
    matInit(Z,size,1.0/n);
    matInit(Y,size,0.0);

    // GPU matrix
    cudaMalloc(&d_C, size*sizeof(float));
    cudaMalloc(&d_X, size*sizeof(float));
    cudaMalloc(&d_Z, size*sizeof(float));
    cudaMalloc(&d_Y, size*sizeof(float));
    
    printf("Copying data from CPU to GPU ...\n");

    // copy to GPU
    cudaMemcpy(d_C, badmm_mt->C, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, Z, sizeof(float)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(float)*size, cudaMemcpyHostToDevice);
    
    if (badmm_mt->a)
    {
        cudaMalloc(&d_a, m*sizeof(float));
        cudaMemcpy(d_a, badmm_mt->a, sizeof(float)*m, cudaMemcpyHostToDevice);
    }
    
    if (badmm_mt->b)
    {
        cudaMalloc(&d_b, m*sizeof(float));
        cudaMemcpy(d_b, badmm_mt->b, sizeof(float)*n, cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&d_Xold, size*sizeof(float));

/*  only support integer
    cudaMemset(d_X,0.0,size*sizeof(float));
    cudaMemset(d_Z,0.1,size*sizeof(float));
    cudaMemset(d_Y,0.0,size*sizeof(float));
*/

    // grid and block size
    unsigned int block_size = size > gpu_info->MAX_BLOCK_SIZE ? gpu_info->MAX_BLOCK_SIZE : size;
    unsigned long int n_blocks = (int) (size+block_size-1)/block_size;
    if(n_blocks > gpu_info->MAX_GRID_SIZE) n_blocks = gpu_info->MAX_GRID_SIZE;
    
    unsigned int stride = block_size*n_blocks;

    cudaMalloc(&d_Yerr, stride*sizeof(float));

    cudaMalloc(&d_rowSum, m*sizeof(float));
    cudaMalloc(&d_colSum, n*sizeof(float));

    // column with all ones
    cudaMalloc(&col_ones, n*sizeof(float));
    unsigned int col_blocksize = n > gpu_info->MAX_BLOCK_SIZE ? gpu_info->MAX_BLOCK_SIZE : n;
    unsigned int n_colblocks = (int) (n+col_blocksize-1)/col_blocksize;
    vecInit<<<n_colblocks,col_blocksize>>>(col_ones,n,1);

    // row with all ones
    cudaMalloc(&row_ones, m*sizeof(float));
    unsigned int row_blocksize = m > gpu_info->MAX_BLOCK_SIZE ? gpu_info->MAX_BLOCK_SIZE : m;
    unsigned int n_rowblocks = (int) (m+row_blocksize-1)/row_blocksize;
    vecInit<<<n_rowblocks,row_blocksize>>>(row_ones,m,1);
    
    thrust::device_ptr<float> dev_ptr, dev_ptr1;
    
    thrust::device_ptr<float> d_Cptr = thrust::device_pointer_cast(d_C);

    printf("nblcoks = %d, block_size = %d, size = %d, stride = %d\n", n_blocks, block_size, size, stride);

    
    printf("BregmanADMM for mass transportation is running ...\n");
    
    cublasInit();

    float iter_obj;
//    float iternorm;

    int iter, count = 0;

    // GPU time
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // d_C = -C/rho
    cublasSscal( size, -1.0/badmmpara->rho, d_C, 1);

    for ( iter = 0; iter < badmmpara->MAX_ITER; iter++ )
    {
        // X update
        cublasScopy(size,d_X,1,d_Xold,1);
        xexp<<<n_blocks,block_size>>>( d_X, d_C, d_Y, d_Z, size);
        cublasSgemv( 'T',n,m, 1.0,d_X,n,col_ones,1, 0,d_rowSum,1);  // fortran, column-major
        if (badmm_mt->a)
            rowNorm_a<<<n_blocks,block_size>>>(d_X, d_rowSum, d_a, size, n);
        else
            rowNorm<<<n_blocks,block_size>>>(d_X, d_rowSum, size, n);

        // Z update
        zexp<<<n_blocks,block_size>>>( d_Z, d_X, d_Y, size);
        cublasSgemv('N',n,m, 1.0,d_Z,n,row_ones,1, 0.0,d_colSum,1);
        if (badmm_mt->b)
            colNorm_b<<<n_blocks,block_size>>>(d_Z, d_colSum, d_b, size, n);
        else
            colNorm<<<n_blocks,block_size>>>(d_Z, d_colSum, size, n);

        // dual update
        dual<<<n_blocks,block_size>>>( d_Yerr, d_Y, d_X, d_Z, size);
        
        // check stopping conditions
        dev_ptr = thrust::device_pointer_cast(d_X);
        dev_ptr1 = thrust::device_pointer_cast(d_Xold);
        Xerr = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, dev_ptr1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr+size, dev_ptr1+size)), zdiffsq(), 0.0f, thrust::plus<float>());
        
        dev_ptr = thrust::device_pointer_cast(d_X);
        // for relative err condition
//        iternorm = thrust::inner_product(dev_ptr, dev_ptr+size, dev_ptr, 0.0f);
//        Xerr = sqrt(Xerr/iternorm);

        dev_ptr = thrust::device_pointer_cast(d_Yerr);
        Yerr = thrust::reduce(dev_ptr, dev_ptr+stride);
        dev_ptr = thrust::device_pointer_cast(d_Y);
        // for relative err condition
//        iternorm = thrust::inner_product(dev_ptr, dev_ptr+size, dev_ptr, 0.0f);
//        Yerr = sqrt(Yerr/iternorm);
        
        if ( Yerr < badmmpara->tol && Xerr < badmmpara->tol ) {
            break;
        }

        if( badmm_mt->print_step && !((iter+1)%badmm_mt->print_step) )
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            // calculate primal objective value
            dev_ptr = thrust::device_pointer_cast(d_Z);
            iter_obj = thrust::inner_product(d_Cptr, d_Cptr+size, dev_ptr, 0.0f);
        
            badmmpara->iter_time[count] = milliseconds;
            badmmpara->iter_err[count] = Xerr + Yerr;
            badmmpara->iter_obj[count] = iter_obj * (-badmmpara->rho);
            count++;

            printf("iter = %d, objval = %f, primal_err = %f, dual_err = %f, time = %f\n", iter, iter_obj * (-badmmpara->rho), Xerr, Yerr, milliseconds);
        }
    }
    // calculate primal objective value
    dev_ptr = thrust::device_pointer_cast(d_Z);
    iter_obj = thrust::inner_product(d_Cptr, d_Cptr+size, dev_ptr, 0.0f);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // average X+Z
//    cublasSaxpy (size, 1, d_Z, 1, d_X, 1);
//    cublasSscal( size, 0.5, d_X, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost,stream);

    badmmpara->iter_err[count] = Xerr + Yerr;
    badmmpara->iter_time[count] = milliseconds;
    badmmpara->iter_obj[count] = iter_obj * (-badmmpara->rho);
    printf("iter = %d, objval = %f, Xerr = %f, Yerr = %f, milliseconds:%f\n", iter, iter_obj * (-badmmpara->rho), Xerr, Yerr, milliseconds);


    if (badmm_mt->SAVEFILE)
    {
        char filename[40];
        FILE *f;
        sprintf(filename, "X_out.dat");
        f = fopen(filename, "wb");
        fwrite (X,sizeof(float),size,f);
        fclose(f);
    }

    cudaFree(d_C);
    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFree(d_Y);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);
    cudaFree(col_ones);
    cudaFree(row_ones);
    cudaFree(d_Yerr);
    cudaFree(d_Xold);
    if (badmm_mt->a)cudaFree(d_a);
    if (badmm_mt->b)cudaFree(d_b);

    
    delete[]X;
    delete[]Z;
    delete[]Y;

    cudaDeviceReset();
}

int main(const int argc, const char **argv)
{

    BADMM_massTrans* badmm_mt = NULL;

    badmm_mt = (struct BADMM_massTrans *) malloc( sizeof(struct BADMM_massTrans) );

    badmm_mt->print_step = 0;           // default: not print
    badmm_mt->SAVEFILE = 1;             // default: save
    badmm_mt->C = NULL;
    badmm_mt->a = NULL;
    badmm_mt->b = NULL;
    
    long size;
    int Asize[2];

    unsigned int dim;

//    dim = 1;
    dim = 5;
//    dim = 10;
//    dim = 15;

    char* str;
    if ( argc > 1 ) dim = strtol(argv[1],&str,10);

    dim = dim*1024;

    // read file
    char filename[40];
    FILE *f;

    // read C
    sprintf(filename, "%dC.dat",dim);
	f = fopen ( filename , "rb" );
    
    if ( f == NULL ) {
        printf("Error! Can not find C file!");
        return 0;
    }

    fread(Asize,sizeof(int),2, f);
    badmm_mt->m = Asize[0];
    badmm_mt->n = Asize[1];
    size = badmm_mt->m*badmm_mt->n;
    badmm_mt->C = new float[size];
    fread (badmm_mt->C,sizeof(float),size,f);
    fclose(f);
    
    printf("Cost Matrix C: rows = %d, cols = %d, total size = %d\n", badmm_mt->m, badmm_mt->n, size);
    
    // read a
    sprintf(filename, "%da.dat",dim);
	f = fopen ( filename , "rb" );
    if ( f != NULL )
    {
        badmm_mt->a = new float[badmm_mt->m];
        fread (badmm_mt->a,sizeof(float),badmm_mt->m,f);
        fclose(f);
    }
    
    // read b
    sprintf(filename, "%b.dat",dim);
	f = fopen ( filename , "rb" );
    if ( f != NULL )
    {
        badmm_mt->b = new float[badmm_mt->n];
        fread (badmm_mt->b,sizeof(float),badmm_mt->n,f);
        fclose(f);
    }

    int iter_size;

    ADMM_para* badmm_para = NULL;
    
    badmm_para = (struct ADMM_para *) malloc( sizeof(struct ADMM_para) );
    
    // default value
    badmm_para->rho = 0.001;
    badmm_para->MAX_ITER = 2000;
    badmm_para->tol = 1e-4;

    if ( argc > 2 ) badmm_para->rho = strtod(argv[2],&str);
    if ( argc > 3 ) badmm_para->MAX_ITER = strtol(argv[3],&str,10);
    if ( argc > 4 ) badmm_para->tol = strtod(argv[4],&str);
    if ( argc > 5 ) badmm_mt->print_step = strtol(argv[5],&str,10);
    if ( argc > 6 ) badmm_mt->SAVEFILE = strtol(argv[6],&str,10);
    
    if ( badmm_para->rho == 0.0 ) badmm_para->rho = 0.001;
    if ( badmm_para->MAX_ITER == 0 ) badmm_para->MAX_ITER = 2000;
    if ( badmm_para->tol == 0.0 ) badmm_para->tol = 1e-4;

    iter_size = 1;
    if(badmm_mt->print_step)
    {
        iter_size = (int)badmm_para->MAX_ITER/badmm_mt->print_step + 1;
    }
    badmm_para->iter_obj = new float[iter_size];
    badmm_para->iter_time = new float[iter_size];
    badmm_para->iter_err = new float[iter_size];
    
    printf("Please be patient! Getting GPU information is slow .....\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);       // default device

    GPUInfo gpu_info;
    gpu_info.MAX_GRID_SIZE = prop.maxGridSize[0];
    gpu_info.MAX_BLOCK_SIZE = prop.maxThreadsPerBlock;

    // if out of GPU memory, return
    float mem = (size*5*4+(badmm_mt->m+badmm_mt->n)*3*4+gpu_info.MAX_GRID_SIZE*gpu_info.MAX_BLOCK_SIZE*2*4)/pow(2,30);
    float GPUmem = (long)prop.totalGlobalMem/pow(2,30);
    printf("gridDim = %d, blockDim = %d, memory required = %fGB, GPU memory = %fGB\n", gpu_info.MAX_GRID_SIZE, gpu_info.MAX_BLOCK_SIZE, mem, GPUmem );
    if ( GPUmem < mem )
    {
        printf("Not enough memory on GPU to solve the problem !\n");
        return 0;
    }

    printf("rho = %f, Max_Iteration = %d, tol = %f, print every %d steps, save result: %d\n", badmm_para->rho, badmm_para->MAX_ITER, badmm_para->tol, badmm_mt->print_step, badmm_mt->SAVEFILE);

    gpuBADMM_MT( badmm_mt, badmm_para, &gpu_info);

    delete[]badmm_para->iter_err;
    delete[]badmm_para->iter_obj;
    delete[]badmm_para->iter_time;
    free(badmm_para);
    if(badmm_mt->C)delete[]badmm_mt->C;
    if(badmm_mt->a)delete[]badmm_mt->a;
    if(badmm_mt->b)delete[]badmm_mt->b;
    free(badmm_mt);
}


void matInit(float* &X, unsigned int size, float value)
{
    for ( int i = 0 ; i < size ; i++ )
        X[i] = value;
}
