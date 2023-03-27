#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>

__global__ void HelloWorld(int* arr){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    arr[index] = 1;
}


int main() {
    size_t size = 20 * sizeof(int);

    int* h_array = (int*)malloc(size);
    __device__ int* d_array;
    cudaMalloc(&d_array, size);

    dim3 threadPerBlock(5, 1, 1);
    dim3 numBlocks(2, 1, 1);
    HelloWorld<<<numBlocks, threadPerBlock>>>(d_array);

    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    cudaFree(d_array);

    for (int i = 0; i < 20; i++){
        printf("%d ", h_array[i]);
    }
    printf("\n");
    free(h_array);

    return 0;
}