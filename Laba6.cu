#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <ctime>
#include <random>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

#define MAXIMUM_THREADS_PER_WARP 32

// <<<block = arr.shape[0], thread = arr.shape[1]>>>
__global__ void init(double* arr){
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(clock64(), tid, 0, &state);
    arr[tid] = curand_uniform_double(&state) * 2 - 1;
}

// <<<block = arr.shape[0], thread = kernel.shape[1]>>>
__global__ void forward_FC(double* arr, double* kernel, double* res){
	size_t i = blockIdx.x;
	size_t size = gridDim.x;
	size_t j = threadIdx.x;
	res[j] += arr[i] * kernel[j * size + i];
}

// <<<block = arr.shape[0], thread = arr.shape[1]>>>
__global__ void sigmoid(double* arr){
	size_t i = blockIdx.x;
	size_t size = blockDim.x;
	size_t j = threadIdx.x;
	arr[i * size + j] = 1.0 / (1.0 + exp(arr[i * size + j]));
}

int main(int argc, char** argv) {
	double *input, *FC1, *FC2, *FC3;

/////////////////////////////////////////////////////////////////////
	// Выделение памяти под входной слой
	cudaMalloc(&input, sizeof(double) * 32 * 32);
	dim3 block = dim3(32);
	dim3 thread = dim3(32);
	init<<<block, thread>>>(input);
/////////////////////////////////////////////////////////////////////
	// Выделение памяти под FC слой
	cudaMalloc(&FC1, sizeof(double) * 32 * 32 * 16 * 16);
	block = dim3(16 * 16);
	thread = dim3(32 * 32);
	// Инициализация весов FC слоя
	init<<<block, thread>>>(FC1);
	// Выделение памяти под результат FC слоя
	double *res_1;
	cudaMalloc(&res_1, sizeof(double) * 16 * 16);
	block = dim3(32 * 32);
	thread = dim3(16 * 16);
	// Проход FC слоем
	forward_FC<<<block, thread>>>(input, FC1, res_1);

	block = dim3(1 * 1);
	thread = dim3(16 * 16);
	sigmoid<<<block, thread>>>(res_1);
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	// Выделение памяти под FC слой
	cudaMalloc(&FC2, sizeof(double) * 16 * 16 * 4 * 4);
	block = dim3(16 * 4);
	thread = dim3(16 * 4);
	// Инициализация весов FC слоя
	init<<<block, thread>>>(FC2);
	// Выделение памяти под результат FC слоя
	double *res_2;
	cudaMalloc(&res_2, sizeof(double) * 4 * 4);
	block = dim3(16 * 16);
	thread = dim3(4 * 4);
	// Проход FC слоем
	forward_FC<<<block, thread>>>(res_1, FC2, res_2);

	block = dim3(1 * 1);
	thread = dim3(4 * 4);
	sigmoid<<<block, thread>>>(res_2);
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	// Выделение памяти под FC слой
	cudaMalloc(&FC3, sizeof(double) * 4 * 4 * 1 * 1);
	block = dim3(1 * 1);
	thread = dim3(4 * 4);
	// Инициализация весов FC слоя
	init<<<block, thread>>>(FC3);
	// Выделение памяти под результат FC слоя
	double *res_3;
	cudaMalloc(&res_3, sizeof(double) * 1 * 1);
	block = dim3(4 * 4);
	thread = dim3(1 * 1);
	// Проход FC слоем
	forward_FC<<<block, thread>>>(res_2, FC3, res_3);
	
	block = dim3(1 * 1);
	thread = dim3(1 * 1);
	sigmoid<<<block, thread>>>(res_3);
/////////////////////////////////////////////////////////////////////

	double *result = new double[1];
	cudaMemcpy(result, res_3, sizeof(double) * 1, cudaMemcpyDeviceToHost);

	std::cout << result[0] << std::endl;

	cudaFree(input);
	cudaFree(FC1);
	cudaFree(FC2);
	cudaFree(FC3);
	cudaFree(res_1);
	cudaFree(res_2);
	cudaFree(res_3);

	return 0;
}