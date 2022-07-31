#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include "./mvncdf_kernel.cu"

constexpr struct {
    unsigned int block_size = 256;
} strcutural;



#define output_ptr(gpu_ptr) ((void **)&gpu_ptr)


struct task_args_t
{
    // number of dimentions of the MVN distribution
    int ndims = 3;

    // number of Monte-Carlo samples
    int nsamples = 1000000;

    // Covariance Matrix
    double C[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    // integration boundaries
    double x0[3] = {0.1, 0.2, 0.0}; // x0

} task_params;


/*
    Runs Monte Carlo integration on GPU
*/
double run_task(task_args_t task_args)
{
    constexpr int TS = sizeof(double);

    int N = task_args.nsamples;
    int m = task_args.ndims;

    double *cpu_ptr_output = (double *)malloc(N * TS);
    double *cpu_ptr_wa = (double *)malloc(N * m * TS);
    double *cpu_ptr_b = (double *)malloc(m * TS);
    double *cpu_ptr_C = (double *)malloc(m * m * TS);


    for(int i = 0; i < m; ++i) {
        cpu_ptr_b[i] = task_args.x0[i];
    }
    for(int i1 = 0; i1 < m; ++i1) {
        for(int i2 = 0; i2 < m; ++i2) {
            cpu_ptr_C[i1 * m + i2] = task_args.C[i1][i2];
        }
    }

    double *gpu_ptr_output; cudaMalloc(output_ptr(gpu_ptr_output), N * TS);
    double *gpu_ptr_wa; cudaMalloc(output_ptr(gpu_ptr_wa), N * m * TS);
    double *gpu_ptr_b; cudaMalloc(output_ptr(gpu_ptr_b), m * TS);
    double *gpu_ptr_C; cudaMalloc(output_ptr(gpu_ptr_C), m * m * TS);

    // todo: fill-in using RNG

    cudaMemcpy(gpu_ptr_wa, /*src*/ cpu_ptr_wa, N*m * TS, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr_b, /*src*/ cpu_ptr_b, m * TS, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ptr_C, /*src*/ cpu_ptr_C, m*m * TS, cudaMemcpyHostToDevice);

    const int block_size = strcutural.block_size;
    const int grid_size = static_cast<int>( N / block_size + 1);

    processOneSample<<<grid_size, block_size>>>(
        gpu_ptr_output,
        gpu_ptr_wa,
        gpu_ptr_b,
        gpu_ptr_C,
        m,
        N);


    // todo: average T s in output
    cudaMemcpy(/*dst*/ cpu_ptr_output, gpu_ptr_output, N * TS, cudaMemcpyDeviceToHost);


    cudaFree(gpu_ptr_output);
    cudaFree(gpu_ptr_wa);
    cudaFree(gpu_ptr_b);
    cudaFree(gpu_ptr_C);

    free(cpu_ptr_output);
    free(cpu_ptr_wa);
    free(cpu_ptr_b);
    free(cpu_ptr_C);

    double T=0.0;
    for(int i=0; i<N;i++) {
        T += cpu_ptr_output[i];
    }
    return T;
}

int main()
{
    double T = run_task(task_params);
    printf("T = %f\n", T);
    return 0;
}