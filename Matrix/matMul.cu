#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#define BlockSize (8)
#define n         (8)

void printMatrix(int *a)
{

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			printf("%d ",a[i*n+j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void matrixMulKernel(int *da, int *db, int *dc)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;

	if((row <n) && (col<n))
		dc[row*n+col]=da[row*n+col]+db[row*n+col];
}

int main(int argc, char **argv)
{
	int *a, *b, *c;
	int size;
	int *da, *db, *dc;
	dim3 block(BlockSize, BlockSize, 1);
	dim3 grid(n/BlockSize, n/BlockSize, 1);
	size_t csize=n*n*sizeof(int);
	cudaError_t err;	

	size=n*n;
	a=(int *)calloc(size, sizeof(int));
	if(a==NULL)
	{
		printf("Error while dynamically allocating matrix A\n");
	}

	b=(int *)calloc(size, sizeof(int));
	if(b==NULL)
	{
		printf("Error while dynamically allocating matrix B\n");
	}

	c=(int *)calloc(size, sizeof(int));
	if(c==NULL)
	{
		printf("Error while dynamically allocating matrix C\n");
	}
	
	for(int i=0; i<size; i++)
	{
		a[i]=2;
		b[i]=2;
	}
	err=cudaMalloc((void **)&da, csize);
	if(err!=cudaSuccess)
	{
		printf("Error while allocating memory for matrix A using cudamalloc\n");
		printf("error string: %s\n",cudaGetErrorString(err));
	}
	if(cudaMemcpy(da, a, size*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("Error in cudaMemCpy while transferring a\n");
		return 0;
	}
	
	if(cudaMalloc((void **)&db, csize)!=cudaSuccess)
	{
		printf("Error while allocating memory for matrix B using cudamalloc\n");
	}
	if(cudaMemcpy(db, b, size*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		printf("Error in cudaMemCpy while transferring b\n");
		return 0;
	}
	
	if(cudaMalloc((void **)&dc, csize)!=cudaSuccess)
	{
		printf("Error while allocating memory for matrix C using cudamalloc\n");
	}
	matrixMulKernel<<<grid, block>>>(da, db, dc);
	if(cudaMemcpy(dc, c, size*sizeof(int), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		printf("Error in cudaMemCpy while transferring c\n");
		return 0;
	}
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	printMatrix(c);

return 0;
}
