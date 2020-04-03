//This file contains code for performing convolution on a GPU
//Author:-Ajay Singh
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int mask_width, int size){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int start_pos=(int)(i-(mask_width/2));
	float pvalue;
	for(int j=start_pos; j<start_pos+mask_width; j++){
		if(j>=0 && j<size)
			pvalue+=M[j-start_pos]*N[j];
	}
	P[i]=pvalue;
}
#define size 10000
#define mask_width 100

int main(){
	
	float mask[mask_width];
	float arr[size];
	float Output[size];
	float *Pd, *arrd, *Md;
	double elapsed;

	for(int i=0; i<size; i++){
		if(i<mask_width){
			mask[i]=i/size;
		}
		arr[i]=(i*i)/size;
	}
	if(cudaMalloc((void **)&Pd, sizeof(float)*size)!=cudaSuccess){
		printf("error while allocating Pd\n");
	}
	
	if(cudaMalloc((void **)&Md, sizeof(float)*mask_width)!=cudaSuccess){
		printf("error while allocating Md\n");
	}
	if(cudaMalloc((void **)&arrd, sizeof(float)*size)!=cudaSuccess){
		printf("error while allocating arrd\n");
	}
	
	if(cudaMemcpy(arrd, arr, sizeof(float)*size, cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("error while copyting arr from host to device\n");
	}
	if(cudaMemcpy(Md, mask, sizeof(float)*mask_width, cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("error while copyting mask from host to device\n");
	}
	elapsed= -clock();
	convolution_1D_basic_kernel<<<size/20, 20>>>(arrd, Md, Pd, mask_width, size);
	elapsed+=clock();
  
  if(cudaMemcpy(Output, Pd, sizeof(float)*size, cudaMemcpyDeviceToHost)!=cudaSuccess){
		printf("error while copyting arr from host to device\n");
	}
	printf("Time Taken = %lf\n",elapsed);
	
	cudaFree(arrd);
	cudaFree(Pd);
	cudaFree(Md);

return 0;
}

