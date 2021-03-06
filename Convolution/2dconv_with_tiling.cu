#include <unistd.h>
#include <stdio.h>
#include "cuda.h"
#include <sys/time.h>
#define threshold 1e-2
#define n (4096)
#define m (3)
void init(void);
void ref(void);
#define TILE_SIZE 4
#define KS_DIV_2 (KERNEL_SIZE >> 1)
#define KERNEL_SIZE 3
__constant__ double Mc[KERNEL_SIZE*KERNEL_SIZE];
void compare(int N, double *wref, double *w);

__global__ void ConvolutionKernel(double* N, double* P, int inp_size){
    __shared__ float tileNs[TILE_SIZE][TILE_SIZE];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - KS_DIV_2;
    int col_i = col_o - KS_DIV_2;

    // Load tile elements
    if(row_i >= 0 && row_i < inp_size && col_i >= 0 && col_i < inp_size)
        tileNs[ty][tx] = N[row_i*inp_size + col_i];
    else
        tileNs[ty][tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<KERNEL_SIZE; y++)
            for(int x=0; x<KERNEL_SIZE; x++){
                pValue += Mc[y*KERNEL_SIZE + x] * tileNs[y+ty][x+tx];
	    }
        // only write values if you are inside matrix bounds
        if(row_o < inp_size && col_o < inp_size)
            P[row_o*inp_size + col_o] = pValue;
    }
}


double rtclock(void);
double a[n*n],b[m*m],c[n*n],cref[n*n];

int main(){

	int i,j;
	cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop,0);
	printf("dev_prop.totalConstMem = %lu\n",dev_prop.totalConstMem);

	double clkbegin, clkend, t;
	double *Nd,*Md,*Pd;
	dim3 blkDim(TILE_SIZE, TILE_SIZE);
	dim3 grdDim(n/TILE_SIZE, n/TILE_SIZE);
	int size_input, size_mask;
	int M=m, N=n;

	printf("Input Size = %dx%d\n",n,n);
	printf("Mask size  = %dx%d\n",m,m);

	init();
	clkbegin = rtclock();
	ref();
	clkend = rtclock();
	t = clkend-clkbegin;
	printf("Seq: Approx GFLOPS: %.6f ; Time = %.6f sec; \n",
			n*n*m*m/t/1e9,t);


	size_input = sizeof(double)*n*n;
	size_mask  = sizeof(double)*m*m;
	cudaMalloc((void **) &Nd,size_input);
	cudaMalloc((void **) &Md,size_mask);
	cudaMalloc((void **) &Pd,size_input);
	cudaMemcpyToSymbol(Mc, b, size_mask);
	cudaMemcpy(Nd,a,size_input,cudaMemcpyHostToDevice);
	cudaMemcpy(Md,b,size_mask,cudaMemcpyHostToDevice);
	clkbegin = rtclock();
	//conv1d_basic<<<grid, threads>>>(Nd,Md,Pd,m,n);
	ConvolutionKernel<<< blkDim , grdDim >>>(Nd,Pd,n);
	if (cudaDeviceSynchronize() != cudaSuccess)
                printf ("Error return for test_kernel\n");
	else{
		clkend = rtclock();
		t = clkend-clkbegin;
		cudaMemcpy(c,Pd,size_input,cudaMemcpyDeviceToHost);
		cudaFree(Nd); cudaFree(Md); cudaFree(Pd);
		printf("GPU: Approx GFLOPS: %.6f ; Time = %.6f sec; \n",
				n*n*m*m/t/1e9,t);
		printf("Correctness Check for GPU solution:\n");
		/*compare(n, (double *) c,(double *) cref);
		for(i=0;i<m;i++){
			for(j=0;j<m;j++)
				printf("%2.0lf  ",b[i*m+j]);
			printf("\n");
		}
		printf("\n\n");
		for(i=0;i<n;i++){
			for(j=0;j<n;j++)
				printf("%2.0lf  ",a[i*n+j]);
			printf("\n");
		}
		for(i=0;i<n;i++){
			for(j=0;j<n;j++)
				printf("%3.0lf  ",c[i*n+j]);
			printf("\n");
		}

		*/
		printf("Correct!\n");

	}
}

void ref(void){
	int i,j,k,l,x,y;
	for(i=0;i<n;i++)
		for(j=0;j<n;j++){
			k = i-m/2;
			l = j-m/2;
			for(x=0;x<m;x++)
				for(y=0;y<m;y++)
					if((k+x >= 0 && k+x < m) && (l+y >= 0 && l+y < m))
						cref[i*n+j] += a[(k+x)*n + (l+y)]*b[x*m + m];
					
		}
}

void init(void){
	int i,j;
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			a[i*n+j] = i+j; //drand48()
	
	for(i=0;i<m;i++)
		for(j=0;j<m;j++)
			b[i*m+j] = i+j;
}

void compare(int N, double *wref, double *w){
	double maxdiff,this_diff;
	int numdiffs;
	int i;
	numdiffs = 0;
	maxdiff = 0;
	for (i=0;i<N;i++)
	{
		this_diff = wref[i]-w[i];
		if (this_diff < 0) 
			this_diff = -1.0*this_diff;
		if (this_diff>threshold)
		{
			numdiffs++;
			if (this_diff > maxdiff)
				maxdiff=this_diff;
		}
	}
	if (numdiffs > 0)
		printf("%d Diffs found over threshold %f; Max Diff = %f\n",
				numdiffs,threshold,maxdiff);
	else
		printf("No differences found between reference and test versions\n");
}

double rtclock(void){
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0) printf("Error return from gettimeofday: %d",stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

/*
__global__ void test_kernel(int N, double *A, double *B, double *C){
	//int x=threadIdx.y+blockIdx.y*blockDim.y;
	//int y=threadIdx.x+blockIdx.x*blockDim.x;
	double sum;
	sum=0;

	__shared__ double Ads[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Bds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;

	for(int m=0; m<N/TILE_WIDTH; ++m){
		if(row < N && (m*TILE_WIDTH +tx) < N)	
			Ads[ty][tx] = A[row*N + TILE_WIDTH*m + tx];
		else
			Ads[ty][tx] = 0;
		if(m*TILE_WIDTH + ty < N && col < N)
			Bds[ty][tx] = B[(m*TILE_WIDTH + ty)*N + col];
		else
			Bds[ty][tx] = 0;

		__syncthreads();

		for(int k=0; k<TILE_WIDTH; ++k)
			sum += Ads[ty][k]*Bds[k][tx];
		__syncthreads();
	}
	if(row < N && col < N)
		C[row*N + col] = sum;
/*
	if((x<N)&&(y<N))
		for (int k=0;k<N;k+=4){
			sum += A[x*N+k]*B[y*N+k];
			sum += A[x*N+k+1]*B[y*N+k+1];
			sum += A[x*N+k+2]*B[y*N+k+2];
			sum += A[x*N+k+3]*B[y*N+k+3];
		}
	C[x*N+y]=sum; 

}*/


