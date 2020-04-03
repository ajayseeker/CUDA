//This file contains a cuda code implementing 2d convolution
//Author: Ajay Singh
#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

#define mask_width (3)
#define mat_size (5)

__constant__ float mask[mask_width];

__global__
void covolution_2d_kernel(float *Mat, float *Ans)
{
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;

	float pvalue=0;
	for(int j=0, i=row; j<mask_width; j++, i++){
		for(int k=col, l=0; l<mask_width; l++, k++){
			pvalue+=Mat[i*mask_width+k]*mask[j*mask_width+l];
		}
	}
	Ans[row*mask_width+col]=pvalue;
	return;
}
	

int main()
{
float *M, *ans, *h_mask;
float *d_M, *d_ans;
int k=mat_size-(mask_width-1);

	h_mask=(float *)malloc(sizeof(float)*mask_width*mask_width);
	M=(float *)malloc(sizeof(float)*mat_size*mat_size);
	ans=(float *)malloc(sizeof(float)*k*k);
	dim3 grid(mask_width, mask_width);	

	if(M==NULL || mask==NULL || ans==NULL){
		printf(" Error while allocating memory in host for mask or the matrix");
		return 0;
	}
	printf("Printing matrix\n");

	for(int i=0; i<mat_size*mat_size; i++){
		M[i]=i+1;
		printf("%lf ", M[i]);
		if(i%mat_size==0 && i!=0)
			putchar('\n');

	}
	printf("\nPrinting Mask\n");
	for(int j=0; j<mask_width*mask_width; j++){
		h_mask[j]=j+1;
		printf("%lf ",h_mask[j]);
		if((j+1)%mask_width==0 && j!=0)
			putchar('\n');
	}
	
	
	if(cudaMalloc( (void **)&d_M, sizeof(float)*mat_size*mat_size)!=cudaSuccess){
		printf("error while allocating memory for matrix on device\n");
	}


	if(cudaMalloc((void **)&d_ans, sizeof(float)*k*k)!=cudaSuccess){
		printf("error while allocating memory for the mask on device\n");
	}

	if(cudaMemcpyToSymbol(mask, h_mask, sizeof(float)*mask_width)!=cudaSuccess){
		printf("error while copying mask from host to constant memory in device\n");
	}

	if(cudaMemcpy(d_M, M, sizeof(float)*mat_size*mat_size, cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("error while copying matrix to device\n");
	}


	covolution_2d_kernel<<< grid, 1>>>(d_M, d_ans);

	if(cudaMemcpy( ans, d_ans, sizeof(float)*k*k, cudaMemcpyDeviceToHost)!=cudaSuccess){
		printf("error while copying mask to device\n");
	}
	printf("\nPrinting ans\n");
	for(int i=0; i<k*k; i++){
		printf("%lf \t",ans[i]);
		if(i%k==0 && i!=0)
			putchar('\n');
	}
	cudaFree(d_M);
	cudaFree(d_ans);
return 0;
}
	

