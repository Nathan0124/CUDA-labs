#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512

#define NUM_BLOCK_ELEMENTS ( BLOCK_SIZE << 1 )


#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)



// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void scan(float * output, float * input, float *aux, int len) 
{
    // Load a segment of the input vector into shared memory

	//printf("%d %d",aux==NULL, len);

    extern __shared__ float scan_array[];
	
    unsigned int thid = threadIdx.x;
	unsigned int start = blockIdx.x * NUM_BLOCK_ELEMENTS;

	//printf("Thread %d, Start %d\n", thid, start);
	
	int ai = thid;
	int bi = thid + BLOCK_SIZE;
	
	// compute spacing to avoid bank conflicts
	unsigned int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	unsigned int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	//printf("A %d %d, B %d %d\n", ai, bankOffsetA, bi, bankOffsetB);
	
	// Cache the computational window in shared memory
    if (start + ai < len)
{
       scan_array[ai + bankOffsetA] = input[start + ai];

	//printf("A Idx %d, value: %f\n", start + ai,  scan_array[ai + bankOffsetA]);
}
    else
       scan_array[ai + bankOffsetA] = 0;
   
    if (start + bi < len)
{
       scan_array[bi + bankOffsetB] = input[start + bi];
	//printf("B Idx %d, value: %f\n", start + bi,  scan_array[bi + bankOffsetB]);
}
    else
       scan_array[bi + bankOffsetB] = 0;

	//printf("copy to shared memory finish\n");

	
    // Reduction
    int stride = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {	
		__syncthreads();
	
	

		if (thid < d)      
        {
            int ai = stride*(2*thid+1)-1;
            int bi = stride*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            scan_array[bi] += scan_array[ai];


		//printf("STRIDE:%d, A %d %f, B %d %f\n", stride, ai, scan_array[ai], bi, scan_array[bi]);
        }
		
		stride <<= 1;
    }

	
	// Store and clear the last elements

	if (thid == 0)
	{
		
		int index = NUM_BLOCK_ELEMENTS - 1;
		//printf("offset: %d\n", CONFLICT_FREE_OFFSET(index));
		//printf("lastidx: %d ->", index);		
	
        	index += CONFLICT_FREE_OFFSET(index);

		//printf("index:%d\n", index);
        
		if (aux!=NULL)
		{

		    // write this block's total sum to the corresponding index in the blockSums array
		    aux[blockIdx.x] = scan_array[index];
		}

		// zero the last element in the scan so it will propagate back to the front
		scan_array[index] = 0;
		//printf("%d\n", index);
	}

	

    // Post reduction
	for (int d = 1; d <= BLOCK_SIZE ; d <<= 1)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            int ai = stride*(2*thid+1)-1;
            int bi = stride*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = scan_array[ai];
            scan_array[ai] = scan_array[bi];
            scan_array[bi] += t;
        }
    }

	__syncthreads();
	

    if (start + ai < len){
       output[start + ai] = scan_array[ai + bankOffsetA];
	//printf("A Idx %d, value: %f -> %f\n", start + ai,  scan_array[ai + bankOffsetA], output[start + ai]);
    }
    if (start + bi < len){
       output[start + bi] = scan_array[bi + bankOffsetB];
	//printf("B Idx %d, value: %f -> %f\n", start + bi,  scan_array[bi + bankOffsetB], output[start + bi]);

     }
} 


__global__ void uniformAdd(float *data, float *aux, int len) 
{
	__shared__ float plus;
	
    unsigned int thid = threadIdx.x;
	if(thid == 0){
		plus = aux[blockIdx.x];
	}
	
	unsigned int ai = blockIdx.x * NUM_BLOCK_ELEMENTS + thid;
	unsigned int bi = ai + BLOCK_SIZE;
	
	__syncthreads();
	
    if (blockIdx.x!=0) {
       if (ai < len)
          data[ai] += plus;
       if (bi < len)
          data[bi] += plus;
    }
}


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.


void prescanArrayRecursive(float * &outArray, float * &inArray, int numElements)
{
	// padding space is used to avoid shared memory bank conflicts
	unsigned int extraSpace = (NUM_BLOCK_ELEMENTS / NUM_BANKS);
	unsigned int sharedMemSize = sizeof(float) * (NUM_BLOCK_ELEMENTS + extraSpace);
	

	if (numElements <= NUM_BLOCK_ELEMENTS)
	{
		//printf("Single block\n");
		//Small array that can be scaned in a single block
		
		dim3 gd(1);
		dim3 bd(BLOCK_SIZE);

		// make sure there are no CUDA errors before we start
    		CUT_CHECK_ERROR("prescanArrayRecursive before kernels");
		//printf("Extra %d\n", extraSpace);
		
		scan<<<gd, bd, sharedMemSize>>>(outArray, inArray, NULL, numElements);

		CUT_CHECK_ERROR("prescan");

		cudaDeviceSynchronize();
	}
	else
	{
		//Large array
		
		//allocate global memory for block sums 
		float *auxArray, *auxArraySums;
		
		int numBlockSums = (int) ceil ((float)numElements / NUM_BLOCK_ELEMENTS );

		//printf("%f\n", (float)numElements / NUM_BLOCK_ELEMENTS );
		//printf("%d elements into %d  blocks\n", numElements, numBlockSums);
		
		CUDA_SAFE_CALL(cudaMalloc((void**) &auxArray, numBlockSums * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void**) &auxArraySums, numBlockSums * sizeof(float)));
		
		
		dim3 gd(numBlockSums);
		dim3 bd(BLOCK_SIZE);
		scan<<<gd, bd, sharedMemSize>>>(outArray, inArray, auxArray, numElements);
		cudaDeviceSynchronize();
		
		//Scan the auxiliary array of block sums recurrently
		prescanArrayRecursive(auxArraySums, auxArray, numBlockSums);
		
		//Add corresponding presums to each block
		uniformAdd<<<gd,bd>>>(outArray, auxArraySums, numElements);
		
		//free auxiliary array
		cudaFree(auxArray);
		cudaFree(auxArraySums);
	}



}


void prescanArray(float* &outArray, float* &inArray, int numElements)
{

	prescanArrayRecursive(outArray, inArray, numElements);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
