#ifndef OPT_KERNEL
#define OPT_KERNEL

#define MAX_SM_THREAD 2048
#define PADDED_INPUT_WIDTH (INPUT_WIDTH + 128)&0xFFFFFF80

void opt_2dhisto(uint32_t *input, size_t width, size_t height, uint32_t *bins);

/* Include below the function headers of any other functions that you implement */
void* AllocOnDevice(size_t size);


uint32_t* CopyInputToDevice(uint32_t **src, size_t y_size, size_t x_size);


void CopyHistToHost(uint32_t *d_data, uint32_t *data, size_t size);


void FreeDevice(uint32_t *data);



#endif
