#include <stdio.h>
const int N = 1000; // can we have a shared const on both host and device?

const float dT = 0.001f;


constexpr struct {
    unsigned int block_size = 256;
} strcutural;

// inplace
__global__ void cuda_simple(float *gpu_ptr, int n)
{
    for (int i = 0; i < n; i++)
    {
        gpu_ptr[i] *= 1.5f;
        gpu_ptr[i] = (float)i;
    }
}

/*
Problem:
    identifier "process1<float, int> " is undefined in device code
*/
// A unit of work, most fine-grained parallelisable unit of execusion:
// inplace
template <typename ElemType, typename SizeT>
void process1(ElemType *gpu_ptr, SizeT i)
{
    gpu_ptr[i] *= 1.5f;
    // gpu_ptr[i] = (float)i;
}


/*
Three classes of arguments:
1. class I. explicit arguments (common value): input args to all kernel executions -- (gpu_ptr, n)
2. class II. implicit instance-specific -- (threadIdx blockIdx)
3. class III. implicit (explicit in call) <<,>> args -- (blockDim. gridDim)

4. `strcutural`: All above are different to `strcutural`. Use to fine-tune hardware (to select and fine-tune class III)
*/
// inplace
__global__ void cuda_process_parallel(float *gpu_ptr, int n)
{
    // class I: (gpu_ptr,n)

    // class II:
    const int xi   = threadIdx.x;
    const int yi   = blockIdx.x;
    // const int zi = gridIdx.x = 0;  // undefined
    // 0
    // 0

    // class III: The <<,>> args
    // threadDim.x = 1
    const int nx = blockDim.x; // 256
    const int ny = gridDim.x;  // 4
    // const int nz = nextDim.x;  // 1  // undefined
    // 1
    // 1

    // ^ These are not the hardware strucutres, but the "call" (execusion/orchestration) structure ie <<,>>

    /*
    Invariants:
        threadIdx.x < blockDim.x = 256
        xi < nx
        blockIdx.x < gridDim.x = 4
        yi < ny
    Meaning:
       (xi, yi)  ∈  256 × 4

       (xi, yi)  ∈  nx × ny
    Generalization:
       0,(xi, yi),0,0, …  ∈  ℝ^[  1 × nx × ny × 1 × … ]
                                  1 × 256 × 4 × 1 × …

    */

    // Coordinate executions: Find your share of execution (your scope)

    // The scope of this (current) kernel / gpu core thread: (also region of interest or gpu memory)
    const int tid = yi * nx + xi;
    /*
    const int tid =
                                                                             yi * nx + xi;
                                                    blockIdx.x * blockDim.x + threadIdx.x;
                               ( (0) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
                  ( ( 0 + gridIdx.x) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    ( ( (0) * nextDim.x + gridIdx.x) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x * 1
    1 * (threadIdx.x +  blockDim.x * (  blockIdx.x + gridDim.x * ( gridIdx.x +  nextDim.x * (0) ) ));
    */

    /*
      Only if single-block (ny==1):
      int begin = xi, stride = nx;
    */

    // `stride`: One level beyond the call <<,>> args : The shortcomings have to be compensated by single kernel-threads using `stride`:
    int stride = nx * ny;
    int begin = tid;

    if (begin < n)
    for (int i = begin; i < n; i += stride)
    {
        // Execute a single logical thread's process
        // process1<float, int>(gpu_ptr, i);
        gpu_ptr[i] *= 1.5f;
    }
}

void cpu_process(float *cpu_ptr, int n)
{
    // inplace
    for (int i = 0; i < n; i++)
    {
        cpu_ptr[i] *= 1.5f;
    }
}

#define output_ptr(gpu_ptr) ((void **)&gpu_ptr)

float sum_elements(float *cpu_ptr, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += dT * cpu_ptr[i];
    }
    return sum;
}

void cpu_print_elements(float *cpu_ptr, int n, char const *prefix_text)
{
    printf("%s ", prefix_text);
    int begin = (n > 0) ? 0 : (N - (-n));
    int end = (n > 0) ? n : N;
    printf("[%d-%d]: ", begin, end);
    for (int i = begin; i < end; i++)
    {
        printf("%f ", cpu_ptr[i]);
    }
    printf("\n");
}

int main()
{
    float *cpu_ptr = (float *)malloc(N * sizeof(float));

    float f = 10.0f;
    float pi2 = 2 * 3.141592653589793238f * f;
    for (int i = 0; i < N; i++)
    {
        float t = i * dT;
        cpu_ptr[i] = sin(t * pi2 * f) * 10000.0f;
    }
    cpu_print_elements(cpu_ptr, -10, "initial    ");
    printf("\n (pre) sum = %f\n", sum_elements(cpu_ptr, N));
    printf("\n (pre) sum = %f\n", sum_elements(cpu_ptr, N));

    float *gpu_ptr = 0;
    const int TS = sizeof(*gpu_ptr);

    cudaMalloc(output_ptr(gpu_ptr), N * TS);
    cudaMemcpy(gpu_ptr, /*src*/ cpu_ptr, N * TS, cudaMemcpyHostToDevice);

    printf("\n (p2) sum = %f\n", sum_elements(cpu_ptr, N));
    printf("cuda\n");

    const int block_size = strcutural.block_size;
    const int grid_size = static_cast<int>( N / block_size + 1);
    /* which is used? (1,1) versus N? */
    cuda_simple<<<grid_size, block_size>>>(gpu_ptr, (int)N);

    printf("\n (p3) sum = %f\n", sum_elements(cpu_ptr, N));

    cpu_print_elements(cpu_ptr, -10, "just before");
    printf("\n (p3b) sum = %f\n", sum_elements(cpu_ptr, N));
    // cpu_process(cpu_ptr,N);
    printf("\n (p3c) sum = %f\n", sum_elements(cpu_ptr, N));
    cpu_print_elements(cpu_ptr, -10, "just after ");
    // cpu_process(cpu_ptr,N);

    printf("\n (p4) sum = %f\n", sum_elements(cpu_ptr, N));

    cudaMemcpy(/*dst*/ cpu_ptr, gpu_ptr, N * TS, cudaMemcpyDeviceToHost);
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
