#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>

#define WIDTH 100;

__global__
void Matrix_multiplication(int *A, int *B, int *C, int n){
	int col=threadIdx.x+blockIdx.x*blockDim.x;
	int row=threadIdx.y+blockIdx.y*blockDim.y;
	int value=0;
	if((row<n)&&(col<n))
		for(int k=0; k<n; k++){
			value+=A[row*n+k]*B[col*k+n];
		}
	C[row*n+col]=value;
}

int main(){
 int *A, *B, *C;
 int *d_A, *d_B, *d_C;
 dim3 dimGrid(10, 10, 1);
 dim3 dimBlock(10, 10, 1);
 
	cudaMallocHost((void**)&A, sizeof(int)*100);
	cudaMallocHost((void**)&B, sizeof(int)*100);
	cudaMallocHost((void**)&C, sizeof(int)*100);
	
	for(int i=0; i<10; i++){
		for(int j=0; j<10; j++){
			A[i*10+j]=i+1;
			B[i*10+j]=1;
		}
	}

	if(cudaMalloc((void **)&d_A, 100*sizeof(int))!=cudaSuccess ){
		printf("cudaMalloc d_A:Error while dynamically allocating memory\n");
		exit(0);
	}
	if(cudaMalloc((void **)&d_B, 100*sizeof(int))!=cudaSuccess){
		printf("cudaMalloc d_B:Error while dynamically allocating memory\n");
		exit(0);
	}
	if(cudaMalloc((void **)&d_C, 100*sizeof(int))!=cudaSuccess){
		printf("cudaMalloc d_C:Error while dynamically allocating memory\n");
		exit(0);
	}
	if((cudaMemcpy(d_A, A, 100*sizeof(int), cudaMemcpyHostToDevice))!=cudaSuccess){
		printf("cudaMemcpy: Error while copying the matrix A\n");
		exit(0);
	}
	if(cudaMemcpy(d_B, B, 100*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess){
		printf("cudaMemcpy: Error while copying the matrix B\n");
		exit(0);
	}
	
	Matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, 100);

	if(cudaMemcpy(d_C, C, 100*sizeof(int), cudaMemcpyDeviceToHost)!=cudaSuccess){
		printf("cudaMemcpy: Error while copying the matrix C\n");
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_B);

	for(int i=0; i<10; i++){
		for(int j=0; j<10; j++){
			printf("%d ",C[i*10+j]);
		}
		printf("\n");
	}
return 0;
}

	
