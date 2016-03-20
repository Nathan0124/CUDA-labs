#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"

/* Include below the implementation of any other functions you need */
void* AllocOnDevice(size_t size)
{
	void* d_data;
	cudaMalloc((void **)&d_data,size);
	cudaMemset(d_data, 0x0, size);
	return d_data;
}

uint32_t* CopyInputToDevice(uint32_t **src, size_t y_size, size_t x_size)
{
	uint32_t* dst;
  	cudaMalloc((void**)&dst,  x_size * y_size * sizeof(uint32_t));

	#pragma unroll
  	for(int i =0 ; i<y_size; i++)
		cudaMemcpy(dst + i * x_size, src[i], x_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	//cudaThreadSynchronize();

	// For debug
	/*
	uint32_t* temp = (uint32_t*)malloc(x_size * y_size * sizeof(uint32_t));
	cudaMemcpy(temp, dst, x_size * y_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for(int i =0; i<y_size; i++)
		for(int j = 0;j<x_size;j++)
		{	
			if(src[i][j]!=temp[i*x_size + j])
				printf("%d %d\n", src[i][j], temp[i*x_size + j]);
		}
	free(temp);
	printf("memery allocation and tranfer finished\n");
	*////

	return dst;
}

// Allocate memery on cuda.
void CopyHistToHost(uint32_t *d_data, uint32_t *data, size_t size)
{
  	cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
}

void FreeDevice(void *data)
{
	cudaFree(data);
	data = NULL;
}




__global__ void hist_smem(uint32_t *input, unsigned int data_size, uint32_t *bins)
{

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x; 

  // subhistogram in each block
  __shared__ uint32_t subHist[HISTO_HEIGHT * HISTO_WIDTH];

	#pragma unroll
  for (int i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH; i += blockDim.x) 
  {
    subHist[i] = 0;
  }

  __syncthreads();

  // process pixels
  // updates our block's partial histogram in global memory
	#pragma unroll
  for (int pos = x; pos < data_size; pos += stride) 
  {
	//printf("#%d of %d\t value is %d\n",pos, data_size, input[pos]);	
	uint32_t value = input[pos];
	atomicAdd(&subHist[value], 1);	
  }
  //printf("sub histogram finished\n");
  __syncthreads();
 	
	#pragma unroll  
  for (int i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH; i += blockDim.x) 
  {
	//printf("%d: %d add %d\n", i, bins[i], subHist[i]);
      	atomicAdd(&bins[i], subHist[i]);
  }
}


// Use set more sub-histogram in each block, then each thread just update one of the 
// sub-histogram, so they would conflict less as divergence increased.
__global__ void hist_smem_tile(uint32_t *input, unsigned int data_size, uint32_t *bins)
{

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x; 

	const size_t N = 6;
	const size_t N_BIN = HISTO_HEIGHT * HISTO_WIDTH;

  	// diverge subhistogram in each block
  	__shared__ uint32_t subHist[N_BIN][N];

	#pragma unroll
  for (int i = threadIdx.x; i < N_BIN * N; i += blockDim.x) 
  {
    subHist[i % N_BIN][i / N_BIN] = 0;
  }

  // process pixels
  // updates our block's partial histogram in global memory
  __syncthreads();

	#pragma unroll
  for (int pos = x; pos < data_size; pos += stride) 
  {

	uint32_t value = input[pos];

	atomicAdd(&subHist[value][threadIdx.x % N], 1);	
  }

  __syncthreads();
 	
	#pragma unroll  
  for (int i = threadIdx.x; i < HISTO_HEIGHT * HISTO_WIDTH; i += blockDim.x) 
  {
	uint32_t sum = 0;
	for(int k = 0; k < N; k++)
		sum += subHist[i][k];

      	atomicAdd(&bins[i], sum);
  }

}


void opt_2dhisto(uint32_t *input, size_t width, size_t height, uint32_t *bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

	dim3 dimBlock(1024);
    	dim3 dimGrid(2048 / 1024 * 8);
	cudaMemset(bins, 0x0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
	//hist_smem<<<dimGrid, dimBlock>>>(input, width * height, bins);
	hist_smem_tile<<<dimGrid, dimBlock>>>(input, width * height, bins);
	cudaThreadSynchronize();

}


