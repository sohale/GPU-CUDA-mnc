#include <stdio.h>
const int N = 1000; // can we have a shared const on both host and device?

__global__ void cuda_hello(float *gpu_ptr)
{
    // printf("Hello World from GPU!\n");
    for (int i = 0; i < N; i++)
    {
        gpu_ptr[i] += 1.1;
    }
}

#define output_ptr(gpu_ptr) ((void **)&gpu_ptr)

float sum_elements(float* cpu_ptr, int n) {
    float sum = 0;
    for(int i = 0; i < n; i++) {
        sum += cpu_ptr[i];
    }
    return sum;
}
int main()
{
    float* cpu_ptr = (float*)malloc(N * sizeof(float));


    for(int i = 0; i < N; i++) {
        float t = i*0.001;
        cpu_ptr[i] = sin(t * 2 * 3.141592653589793238);
    }
    printf("(pre) sum = %f\n", sum_elements(cpu_ptr, N));



    float *gpu_ptr = 0;
    const int TS = sizeof(*gpu_ptr);
    // const int N = 1000;
    cudaMalloc(output_ptr(gpu_ptr), N * TS);
    cudaMemcpy(gpu_ptr, /*src*/cpu_ptr, N * TS, cudaMemcpyDeviceToHost);

       /* which is used? (1,1) versus N? */
       cuda_hello<<<1, 1>>>(gpu_ptr);

    cudaMemcpy(/*dst*/cpu_ptr, gpu_ptr, N * TS, cudaMemcpyDeviceToHost);
    cudaFree(gpu_ptr);

    float sum = sum_elements(cpu_ptr, N);
    free(cpu_ptr);

    printf("(post) sum = %f\n", sum);
    return 0;
}
