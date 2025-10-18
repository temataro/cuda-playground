// nvcc -o vector_add vector_add.c
// (make vector_add)
#include <stdio.h>
#include "utils.cu"

// This is how you define kernels
__global__ void add(int *a, int *b, int *c, int N)
{
    int index = threadIdx.x;
    if (index < N)
    {
        c[index] = a[index] + b[index];
    }
}
// you call kernels with `<<<function_name>>>(*function_parameters);`

int main(void)
{
    get_device_properties();
    int N = 512; // This is the size of the vector
    int size = N * sizeof(int); // Store N integers in memory

    /* mallocs have void vectors that we're typecasting into integer because...
       reasons.
    */
    int *h_a = (int *)malloc(size); // Allocate memory for vector of addends
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size); // Allocate memory for vector of sums

    // initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = 1024 - i;
    }
    // std::print("Testing {:05.2f}", 2 ^ 9.2);

    /*
      https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions
       cudaMalloc
     */

    int *d_a, *d_b, *d_c; // device memory allocation
    cudaMalloc((void **)&d_a, size); // A pointer to a pointer (that points to
                                     // the first element of the array)
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy to device -- the gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_c, size, cudaMemcpyHostToDevice);

    // run threads
    // clang-format off
    add<<<1, N>>>(d_a, d_b, d_c, N);
    // clang-format on

    // Copy back to host -- the cpu
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%d,", h_c[i]);
    }

    // Free all those mallocs
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
