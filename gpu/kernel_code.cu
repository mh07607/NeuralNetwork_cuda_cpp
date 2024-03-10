#include "kernel_code.h"

__global__ void DenseForwardPass(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P) {
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < P; ++k) {
			Pvalue += d_M[Row*P+k]*d_N[k*N+Col];
		}
		d_P[Row*N+Col] = Pvalue + d_B[Row*N+Col];
	}
}

__global__ void test(){
	int x = 5;
}

__global__ void DenseBackwardPass(){
	test<<<1, 1>>>();
}

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, float * d_B, int M, int N, int P) {
	// Calculate the row index of the d_Pelement and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column index of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < M) && (Col < N)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < P; ++k) {
			Pvalue += d_M[Row*P+k]*d_N[k*N+Col];
		}
		d_P[Row*N+Col] = Pvalue;
	}
}


// __global__ void MatrixSubtractionKernel(float* d_M, float* d_N, float* d_P){
// 	int Row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int Col = blockIdx.x * blockDim.x + threadIdx.x;

// 	if((Row < M) && (Col < N)){
// 		d_P[Row*N+Col] = ;
// 	}
// }
