#include <stdio.h>
const int N = 1000; // can we have a shared const on both host and device?

const float dT = 0.001f;

__global__ void cuda_simple(float *gpu_ptr, int n)
{
    // inplace
    // printf("Hello World from GPU!\n");
    for (int i = 0; i < n; i++)
    {
        // gpu_ptr[i] *= 1.5;
        gpu_ptr[i] = i;
    }
}

// inplace
template<typename ElemType, typename SizeT>
void process1(ElemType * gpu_ptr, SizeT n) {
    for (SizeT i = 0; i < n; i++)
    {
        // gpu_ptr[i] *= 1.5;
        gpu_ptr[i] = i;
    }
}

// sample input args to all kernel executions
__global__ void cuda_process_parallel(float *gpu_ptr, int n)
{
    // inplace

    // (xi, yi)  ∈  nx × ny
    const xi = threadIdx.x;
    const yi = blockIdx.x;
    const nx = blockDim.x;  // 256
    const ny = gridDim.x;   // 4?

    // threadIdx.x < blockDim.x = 256
    // blockIdx.x < gridDim.x = 4 (?)
    // xi < nx
    // yi < ny
    // (xi, yi)  ∈  256 × 4

    // const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = yi * nx + xi;

    // if (tid < n) {}
    // if (ny==1):
    int stride = nx;
    int begin = xi;
    for (int i = begin; i < n; i+=stride)
    {
        // gpu_ptr[i] *= 1.5;
        gpu_ptr[i] = i;
    }

    // process1(gpu_ptr, n);
}

void cpu_process(float *cpu_ptr, int n)
{
    // inplace
    for (int i = 0; i < n; i++)
    {
        cpu_ptr[i] *= 1.5;
    }
}

#define output_ptr(gpu_ptr) ((void **)&gpu_ptr)

float sum_elements(float* cpu_ptr, int n) {
    float sum = 0;
    for(int i = 0; i < n; i++) {
        sum += dT * cpu_ptr[i];
    }
    return sum;
}

void cpu_print_elements(float *cpu_ptr, int n, char const * prefix_text) {
    printf("%s ", prefix_text);
    int begin = (n > 0) ? 0 : (N-(-n));
    int end   = (n > 0) ? n : N;
    printf("[%d-%d]: ", begin, end);
    for(int i = begin; i < end; i++) {
       printf("%f ", cpu_ptr[i]);
    }
    printf("\n");
}

int main()
{
    float* cpu_ptr = (float*)malloc(N * sizeof(float));

    float f = 10.0;
    float pi2 = 2 * 3.141592653589793238 * f;
    for(int i = 0; i < N; i++) {
        float t = i * dT;
	cpu_ptr[i] = sin(t * pi2 * f) * 10000.0;
    }
    cpu_print_elements(cpu_ptr, -10, "initial    ");
    printf("\n (pre) sum = %f\n", sum_elements(cpu_ptr, N));
    printf("\n (pre) sum = %f\n", sum_elements(cpu_ptr, N));

    float *gpu_ptr = 0;
    const int TS = sizeof(*gpu_ptr);

    cudaMalloc(output_ptr(gpu_ptr), N * TS);
    cudaMemcpy(gpu_ptr, /*src*/cpu_ptr, N * TS, cudaMemcpyHostToDevice);

       printf("\n (p2) sum = %f\n", sum_elements(cpu_ptr, N));
       printf("cuda\n");

       /* which is used? (1,1) versus N? */
       cuda_simple<<<1, 1>>>(gpu_ptr, (int)N);

       printf("\n (p3) sum = %f\n", sum_elements(cpu_ptr, N));

       cpu_print_elements(cpu_ptr, -10, "just before");
       printf("\n (p3b) sum = %f\n", sum_elements(cpu_ptr, N));
       // cpu_process(cpu_ptr,N);
       printf("\n (p3c) sum = %f\n", sum_elements(cpu_ptr, N));
       cpu_print_elements(cpu_ptr, -10, "just after ");
       // cpu_process(cpu_ptr,N);

       printf("\n (p4) sum = %f\n", sum_elements(cpu_ptr, N));

    cudaMemcpy(/*dst*/cpu_ptr, gpu_ptr, N * TS, cudaMemcpyDeviceToHost);
    cpu_print_elements(cpu_ptr, -10, "end1        ");
    printf("\n (p5) sum = %f\n", sum_elements(cpu_ptr, N));
    cudaFree(gpu_ptr);

    cpu_print_elements(cpu_ptr, -10, "end2        ");
    float sum = sum_elements(cpu_ptr, N);
    printf("\n (p) sum = %f\n", sum_elements(cpu_ptr, N));
    cpu_print_elements(cpu_ptr, -10, "end3       ");
    free(cpu_ptr);
    cpu_print_elements(cpu_ptr, -10, "end4       ");

    printf("(post) sum = %f\n", sum);
    return 0;
}
