#include <stdio.h>
const int N = 1000; // can we have a shared const on both host and device?

__global__ void cuda_hello(float* gpu_ptr){
    //printf("Hello World from GPU!\n");
    for(int i=0;i<N;i++) {
	    gpu_ptr[i] += 1.1;
    }
}

#define output_ptr(gpu_ptr) ((void**)&gpu_ptr)

int main() {
	float* gpu_ptr = 0;
	const int TS = sizeof(*gpu_ptr);
	// const int N = 1000;
        cudaMalloc( output_ptr(gpu_ptr) , N*TS);
    	cuda_hello<<<1,1>>>(gpu_ptr);

        cudaFree(gpu_ptr);	
    return 0;
}

